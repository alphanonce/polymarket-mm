import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Paper Trading Dashboard',
  description: 'Real-time paper trading dashboard for Polymarket',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-gray-900 text-gray-100 min-h-screen">
        <nav className="border-b border-gray-800 px-6 py-4">
          <div className="flex items-center justify-between max-w-7xl mx-auto">
            <h1 className="text-xl font-bold">Paper Trading Dashboard</h1>
            <div className="text-sm text-gray-400">
              Polymarket Market Making
            </div>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-6 py-8">
          {children}
        </main>
      </body>
    </html>
  )
}
