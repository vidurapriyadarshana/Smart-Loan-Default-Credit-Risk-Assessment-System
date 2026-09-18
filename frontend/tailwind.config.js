/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        fintech: {
          dark: '#090d16',
          surface: '#111827',
          card: 'rgba(17, 24, 39, 0.8)',
          cardHover: 'rgba(31, 41, 55, 0.9)',
          border: 'rgba(255, 255, 255, 0.08)',
          primary: '#0ea5e9',
          accent: '#38bdf8'
        }
      },
      fontFamily: {
        heading: ['Outfit', 'sans-serif'],
        sans: ['Inter', 'sans-serif']
      }
    },
  },
  plugins: [],
};
