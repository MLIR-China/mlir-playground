/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack: (config, options) => {
    config.module.rules.push({
      test: /\.wasm/,
      type: "javascript/auto",
      loader: "file-loader",
      options: {
        publicPath: "public",
      },
    });
    return config
  },
}

module.exports = nextConfig
