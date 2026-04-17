import { NextResponse } from 'next/server';

const API = process.env.API_URL ?? 'http://localhost:8000';

export async function GET() {
  try {
    const upstream = await fetch(`${API}/minerals/scenarios`, { cache: 'no-store' });
    const data = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (e) {
    return NextResponse.json(
      { detail: e instanceof Error ? e.message : 'Proxy error' },
      { status: 502 }
    );
  }
}
