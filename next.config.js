/** @type {import('next').NextConfig} */

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
    isProduction: process.env.NODE_ENV === "production",
    productionDomain: process.env.PRODUCTION_DOMAIN,
    // Prefix URL for generated wasm files.
    wasmGenPrefix: process.env.WASM_GEN_PREFIX,
    // API endpoint for generating quick share links.
    // If empty, quick share link button will not appear.
    shareLinkGenerator: process.env.SHARE_LINK_GENERATOR,
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
    return config;
  },
};

module.exports = nextConfig;
