import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'dscache.tencent-cloud.cn',
        port: '',
        pathname: '/**',
      },
    ],
  },
};

export default nextConfig;
