/** @type {import('next').NextConfig} */
const isProd = process.env.NODE_ENV === 'production';

const nextConfig = {
  reactStrictMode: true,
  images: {
    loader: "custom",
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    nextImageExportOptimizer: {
      imageFolderPath: "public/images",
      exportFolderPath: "out",
      quality: 75,
    },
  },
  env: {
    storePicturesInWEBP: true,
    staticFilePrefix: isProd ? 'https://static.mlir-china.org/' : '',
  },
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
