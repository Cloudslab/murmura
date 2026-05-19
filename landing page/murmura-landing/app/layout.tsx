import type { Metadata } from 'next'
import './globals.css'
import React from "react";

export const metadata: Metadata = {
  title: 'Murmura - Modular Decentralized Federated Learning',
  icons: ['/favicon.png'],
  description: 'An open-source, config-driven framework for Byzantine-resilient decentralized federated learning. Swap aggregators, topologies, and backends from a single YAML file.',
  keywords: ['federated learning', 'decentralized', 'byzantine resilient', 'open source', 'modular', 'federated learning framework', 'aggregation algorithms', 'network topology'],
  authors: [
    { name: 'Murtaza Rangwala' },
    { name: 'Richard O. Sinnott' },
    { name: 'Rajkumar Buyya' }
  ],
  openGraph: {
    title: 'Murmura - Modular Decentralized Federated Learning',
    description: 'An open-source, config-driven framework for Byzantine-resilient decentralized federated learning.',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Murmura - Modular Decentralized Federated Learning',
    description: 'An open-source, config-driven framework for Byzantine-resilient decentralized federated learning.',
  },
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
