# nanami233.com 静态部署

本站使用 Cloudflare Pages 的 Git 自动部署，生产分支为 main。

## 构建设置

- 框架预设：Astro
- 根目录：仓库根目录（留空或 /）
- 构建命令：`pnpm run build:pages`
- 输出目录：`dist`
- NODE_VERSION：`24.22.1`
- PNPM_VERSION：`11.28.2`

Pages 不需要填写 `wrangler deploy` 部署命令。本次修改合并到远端后才能使用新增的 build:pages 命令。

## 静态部署检查

`build:pages` 在构建后拒绝根目录 functions、dist/functions 和 dist/_worker.js，确保产物没有 Pages Functions 或 Worker 入口。仅访问静态页面不会触发新构建。

Cloudflare 的纯静态 Pages 请求免费且不限请求次数；Functions 有独立配额。此检查防止误部署运行时代码，不代表已经配置 WAF 或能保证绝无攻击。见 [官方计费说明](https://developers.cloudflare.com/pages/functions/pricing/)。

## 自定义域名与更新

在 Pages 项目的“自定义域”中绑定 nanami233.com，按控制台要求完成 DNS 和证书验证。Git 推送成功后检查生产部署状态，再访问域名确认内容。旧的 Workers Hello World 如仍绑定同一域名，需要在控制台核对路由。

## 当前内容

站点默认简体中文，博主 nanami。示例文章、动态、系列、相册、歌单及展示数据已清空。尚无内容的扩展栏目暂时关闭，可在 src/config 中重新开启并填入自己的数据。

主题横幅、头像、图标及必要字体保留；字体随站点托管，示例音乐和外部歌单关闭。浏览器缓存资源由 public/_headers 配置。
