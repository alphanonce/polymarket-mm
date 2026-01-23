/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Custom colors for trading dashboard
        profit: '#22c55e',
        loss: '#ef4444',
        neutral: '#6b7280',
      },
    },
  },
  plugins: [],
}
