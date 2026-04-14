import { Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/AppShell";
import DashboardPage from "./pages/DashboardPage";
import LearningPage from "./pages/LearningPage";
import OnboardingPage from "./pages/OnboardingPage";
import { useLearningStore } from "./store/useLearningStore";

export default function App() {
  const state = useLearningStore((state) => state);
  const userInput = state.userInput;
  const prediction = state.prediction;
  const isAnalyzed = Boolean(userInput && prediction);

  console.log("App loaded");
  console.log("State:", state);

  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Navigate to="/onboarding" replace />} />
        <Route path="/onboarding" element={<OnboardingPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/learn/:itemId" element={isAnalyzed ? <LearningPage /> : <Navigate to="/onboarding" replace />} />
        <Route path="*" element={<Navigate to="/onboarding" replace />} />
      </Routes>
    </AppShell>
  );
}
