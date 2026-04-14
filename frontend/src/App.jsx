import { Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/AppShell";
import DashboardPage from "./pages/DashboardPage";
import LearningPage from "./pages/LearningPage";
import OnboardingPage from "./pages/OnboardingPage";
import { useLearningStore } from "./store/useLearningStore";

export default function App() {
  const userInput = useLearningStore((state) => state.userInput);
  const prediction = useLearningStore((state) => state.prediction);
  const isAnalyzed = Boolean(userInput && prediction);

  return (
    <AppShell>
      <Routes>
        <Route path="/onboarding" element={<OnboardingPage />} />
        <Route path="/dashboard" element={isAnalyzed ? <DashboardPage /> : <Navigate to="/onboarding" replace />} />
        <Route path="/learn/:itemId" element={isAnalyzed ? <LearningPage /> : <Navigate to="/onboarding" replace />} />
        <Route path="*" element={<Navigate to={isAnalyzed ? "/dashboard" : "/onboarding"} replace />} />
      </Routes>
    </AppShell>
  );
}
