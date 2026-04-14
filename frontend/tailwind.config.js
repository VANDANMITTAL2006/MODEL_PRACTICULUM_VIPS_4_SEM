/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Sora", "sans-serif"],
        body: ["Manrope", "sans-serif"],
      },
      colors: {
        ink: "#101422",
        sunrise: "#FF7A18",
        sky: "#4AA3FF",
        mint: "#29C48A",
      },
      boxShadow: {
        card: "0 16px 50px -18px rgba(16, 20, 34, 0.42)",
      },
      keyframes: {
        rise: {
          "0%": { opacity: "0", transform: "translateY(18px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        rise: "rise 450ms ease-out both",
      },
    },
  },
  plugins: [],
};
