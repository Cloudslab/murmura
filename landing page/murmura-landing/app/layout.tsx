import type { Metadata } from 'next'
import './globals.css'
import React from "react";

export const metadata: Metadata = {
  title: 'Murmura - Evidential Trust-Aware Decentralized Federated Learning',
  icons: ['/favicon.png'],
  description: 'A modular framework for Evidential Trust-Aware Decentralized Federated Learning with Byzantine-resilient aggregation for Wearable IoT. Leverages Dirichlet-based uncertainty decomposition for intelligent peer evaluation.',
  keywords: ['federated learning', 'decentralized', 'byzantine resilient', 'wearable IoT', 'evidential deep learning', 'trust-aware', 'uncertainty quantification', 'activity recognition'],
  authors: [
    { name: 'Murtaza Rangwala' },
    { name: 'Richard O. Sinnott' },
    { name: 'Rajkumar Buyya' }
  ],
  openGraph: {
    title: 'Murmura - Evidential Trust-Aware Decentralized Federated Learning',
    description: 'A modular framework for Byzantine-resilient decentralized FL with uncertainty-driven peer evaluation for Wearable IoT.',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Murmura - Evidential Trust-Aware Decentralized Federated Learning',
    description: 'Byzantine-resilient decentralized FL with uncertainty-driven peer evaluation for Wearable IoT.',
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
