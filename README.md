# 喜马拉雅 · Music Assistant

为 Music Assistant 接入中国版喜马拉雅，在 MA 中浏览、搜索和收听有声书、播客及电台。

## 主要功能

- **扫码登录**：使用中国版喜马拉雅 App 扫码授权，支持多个喜马拉雅账号。
- **内容搜索**：搜索专辑、声音和主播，查看主播作品及内容详情。
- **分类浏览**：浏览订阅、最近收听、电台广播、热播榜单、猜你喜欢和详细分类。
- **有声书章节**：以整本有声书展示章节，支持章节切换、定位和断点续播。
- **音质选择**：提供最高可用、高清和标准档位，按账号权限及实际资源选择。
- **订阅同步**：可选同步喜马拉雅订阅与 MA 媒体库，支持配置检查间隔。
- **播放进度**：可选同步喜马拉雅云端收听进度。
- **跳过片头片尾**：读取并执行喜马拉雅 App 中为专辑设置的跳过时长。
<img width="1918" height="1042" alt="PixPin_2026-10-18_16-40-46" src="https://github.com/user-attachments/assets/7c42e01d-0827-4871-ba81-fee7e3ab75d4" />
<img width="825" height="687" alt="PixPin_2026-10-18_16-10-39" src="https://github.com/user-attachments/assets/e3d1eb84-beb6-4a3a-adea-9d88261a80bb" />
<img width="1902" height="815" alt="PixPin_2026-10-18_16-42-29" src="https://github.com/user-attachments/assets/3c14fd89-7899-4060-81b2-051521f08dd2" />


## 注意事项

插件需搭配喜马拉雅 API 服务；账号登录和功能设置均在 MA 内完成。

同一喜马拉雅账号下，同一本书仅保留最后一次成功上报的播放位置进度，注意：不同 MA 用户使用同一喜马拉雅账号时将共享进度。需要独立进度时，可使用不同喜马拉雅账号。

章节定位和片头片尾跳过依据喜马拉雅提供的目录时长，目录时长不准确时可能出现不跳过片尾或提前跳过片尾。

目前不提供有声书歌词、实时文稿或文本显示。

仅支持中国版喜马拉雅，内容可用性取决于账号授权；不支持国际版 Himalaya 或付费权限绕过。

## 使用方法

将 music-assistant-ximalaya/providers/ximalaya/ 文件夹放置于 music assistant 的 providers 文件夹内。
docker安装 [bassce/ximalaya-api](https://hub.docker.com/r/bassce/ximalaya-api/) 项目。
