/** @type {import('next').NextConfig} */
const apiUrl = process.env.API_URL || 'https://causal-engine.fly.dev';

const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
