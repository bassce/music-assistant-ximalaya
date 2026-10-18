"""All configuration stays in MA; the headless service owns upstream authorization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
from pathlib import Path

# ruff: noqa: RUF001 -- Chinese messages and punctuation are intentional.
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import aiohttp
from music_assistant.models.setup_flow import SetupFlowError, StepExpiredError
from music_assistant_models.enums import ConfigEntryType

from .client import DEFAULT_API_URL, CarClient, QrExpired, api_url
from .labels import config_entry
from .protocol import CarError

if TYPE_CHECKING:
    from music_assistant.models.setup_flow import SetupSession

QR_WINDOW = 180
_SETUP_LOCKS: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = WeakKeyDictionary()


async def _finish_account(session: SetupSession, client: CarClient) -> None:
    """Serialize account checks and creation, including simultaneous QR confirmations."""
    loop = asyncio.get_running_loop()
    lock = _SETUP_LOCKS.setdefault(loop, asyncio.Lock())
    async with lock:
        # Reauthorization is already bound to the original UID by poll_qr. Allow
        # existing installations to renew, including pre-existing duplicates.
        if not session.context.instance_id:
            uid = str(client.token["uid"])
            configs = await session.mass.config.get_provider_configs(provider_domain="ximalaya")
            for config in configs:
                raw = session.mass.config.get_provider_setup_value(
                    config.instance_id, "credentials"
                )
                try:
                    old = json.loads(str(raw or "{}"))
                    old_uid = str(old.get("uid") or (old.get("token") or {}).get("uid", ""))
                except ValueError, TypeError, AttributeError:
                    raise SetupFlowError(
                        "已有喜马拉雅实例的账号信息无法校验，请先检查该实例"
                    ) from None
                if old_uid == uid:
                    raise SetupFlowError(
                        "此喜马拉雅账号已添加，请使用已有实例，或用另一个账号扫码登录"
                    )
        if commit := getattr(client, "commit_login", None):
            await commit()
        await session.finish({"credentials": json.dumps(client.export())})


def _connection_key(session: SetupSession, base_url: str, supplied: str | None = None) -> str:
    config = session.mass.config
    storage_key = "ximalaya_api_connections/" + hashlib.sha256(base_url.encode()).hexdigest()
    if supplied:
        key = supplied
    elif stored := config.get(storage_key):
        key = config.decrypt_string(stored)
    else:
        key = secrets.token_urlsafe(32)
    # Persist before pairing so cancellation or a lost response cannot strand the backend.
    config.set(storage_key, config.encrypt_string(key), immediate=True)
    return str(key)


async def run_setup(session: SetupSession) -> None:  # noqa: PLR0915 -- Keep login commit and cancellation under one setup lifetime.
    """Configure the backend and complete QR sign-in from MA."""
    try:
        old = json.loads(str(session.context.setup_data.get("credentials") or "{}"))
        if not isinstance(old, dict):
            raise TypeError
    except ValueError, TypeError:
        raise SetupFlowError("已有账号配置无法读取，请检查当前实例") from None
    expected_uid = old.get("uid") or (old.get("token") or {}).get("uid")
    if session.context.instance_id and not expected_uid:
        raise SetupFlowError("已有账号缺少身份信息，无法安全迁移")
    prefill = old.get("api_url", DEFAULT_API_URL)
    errors = None
    async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar(), trust_env=False) as http:
        while True:
            values = await session.form(
                [
                    config_entry(
                        key="api_url",
                        type=ConfigEntryType.STRING,
                        default_value=DEFAULT_API_URL,
                        value=prefill,
                    ),
                    config_entry(
                        key="service_key",
                        type=ConfigEntryType.SECURE_STRING,
                        required=False,
                    ),
                    config_entry(
                        key="reauthorize",
                        type=ConfigEntryType.BOOLEAN,
                        default_value=not bool(expected_uid),
                    ),
                ],
                step_id="api_connection",
                errors=errors,
                last_step=False,
            )
            client = None
            try:
                base_url = api_url(str(values["api_url"]))
                prefill = base_url
                key = _connection_key(
                    session,
                    base_url,
                    str(
                        values.get("service_key")
                        or (old.get("service_key") if old.get("api_url") == base_url else "")
                        or ""
                    )
                    or None,
                )
                credentials = {"api_url": base_url, "service_key": key}
                if old.get("api_url") == base_url:
                    credentials.update(
                        {
                            k: old[k]
                            for k in ("account_id", "account_key", "uid", "nickname")
                            if k in old
                        }
                    )
                client = CarClient(http, credentials)
                try:
                    await client.request("GET", "/v1/capabilities", owner=True)
                except CarError:
                    await client.request("POST", "/v1/pair", {"service_key": key}, anonymous=True)
                capabilities = await client.request("GET", "/v1/capabilities", owner=True)
                if 1 not in capabilities.get("supported_api_versions", []):
                    raise CarError("外置 API 版本不兼容")
                if old.get("token") and not values.get("reauthorize"):
                    result = await client.request(
                        "POST", "/v1/accounts/import", {"credentials": old}, owner=True
                    )
                    if str(result["uid"]) != str(expected_uid):
                        raise CarError("迁移账号校验失败")
                    client.credentials.update(result)
                    client.token = {"uid": result["uid"]}
                    await _finish_account(session, client)
                    return
                if client.credentials.get("account_key") and not values.get("reauthorize"):
                    await client.start()
                    await _finish_account(session, client)
                    return
                if not capabilities.get("client_configured"):
                    config = old.get("client")
                    if not config:
                        path = Path(session.mass.storage_path) / "ximalaya-client.json"
                        if await asyncio.to_thread(path.is_file):
                            config = json.loads(await asyncio.to_thread(path.read_text))
                    if not config:
                        supplied = await session.form(
                            [
                                config_entry(
                                    key="client_json",
                                    type=ConfigEntryType.SECURE_STRING,
                                    required=True,
                                ),
                            ],
                            step_id="client_config",
                            last_step=False,
                        )
                        config = json.loads(str(supplied["client_json"]))
                    await client.request("PUT", "/v1/client", {"client": config}, owner=True)
                for _ in range(3):
                    qr_id, state, image = await client.create_qr()
                    try:
                        await session.progress_until(
                            client.poll_qr(qr_id, state, expected_uid),
                            step_id="scan_qr",
                            text="请使用中国版喜马拉雅 App 扫码并确认登录",
                            image=image,
                            expires_in=QR_WINDOW,
                        )
                    except StepExpiredError, QrExpired:
                        await client.cancel_login()
                        continue
                    await _finish_account(session, client)
                    return
                raise CarError("等待扫码超时，请重新生成二维码")
            except (CarError, SetupFlowError) as err:
                errors = {"base": str(err)}
            except ValueError, TypeError, KeyError, OSError:
                errors = {"base": "配置格式无效，请检查服务地址或客户端 JSON"}
            finally:
                if client:
                    await client.cancel_login()
