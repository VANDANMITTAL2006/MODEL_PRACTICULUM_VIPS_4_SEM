# PulseLearn AI Frontend (React)

Production-grade React frontend for adaptive recommendations.

## Features

- Onboarding flow for cold-start profile capture
- Dashboard with three recommendation lanes:
  - Recommended For You
  - Continue Learning
  - Explore New Topics
- Learning page with timer-based interaction tracking
- Feedback events (`like`, `dislike`, `skip`, `click`, `complete`)
- Real-time recommendation refresh after each interaction
- Optimistic UI updates and analytics session logging

## Stack

- React + hooks
- Zustand state store
- Tailwind CSS
- Axios service layer
- Framer Motion animations

## Backend Integration

Primary endpoints:

- `POST /recommend-content`
- `POST /feedback-event`

Compatibility fallback support:

- `GET /recommend`
- `POST /feedback`

Set API URL in `.env`:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## Run

```bash
npm install
npm run dev
```

Open `http://localhost:5173`.
