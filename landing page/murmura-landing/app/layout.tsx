import type { Metadata } from 'next'
import './globals.css'
import React from "react";

export const metadata: Metadata = {
  title: 'Murmura',
  description: 'Murmura is a framework for federated learning in decentralized/P2P environments. It helps researchers and developers experiment with distributed machine learning while maintaining privacy.',
  generator: 'v0.dev',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
    <body suppressHydrationWarning={true}>{children}</body>
    </html>
  )
}
