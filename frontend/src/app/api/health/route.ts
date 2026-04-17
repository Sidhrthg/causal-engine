import { NextResponse } from 'next/server';

const API = process.env.API_URL ?? 'http://localhost:8000';

export async function GET() {
  try {
    const upstream = await fetch(`${API}/health`, { cache: 'no-store' });
    const data = await upstream.json();
    return NextResponse.json({ ...data, backend: API }, { status: upstream.status });
  } catch {
    return NextResponse.json({ status: 'backend_unreachable', backend: API }, { status: 200 });
  }
}
