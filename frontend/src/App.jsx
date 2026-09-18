import React, { useState, useEffect } from 'react';
import { 
  Building2, Activity, Sparkles, Send, RefreshCw, 
  TrendingUp, Percent, DollarSign, UserCheck, ShieldAlert 
} from 'lucide-react';
import RiskGauge from './components/RiskGauge';
import MetricCard from './components/MetricCard';
import RiskFactorsList from './components/RiskFactorsList';

const API_BASE = 'http://localhost:8000';

export default function App() {
  // Form State
  const [formData, setFormData] = useState({
    person_age: 28,
    person_income: 65000,
    person_home_ownership: 'RENT',
    person_emp_length: 4.0,
    loan_intent: 'PERSONAL',
    loan_grade: 'B',
    loan_amnt: 10000,
    loan_int_rate: 11.2,
    cb_person_default_on_file: 'N',
    cb_person_cred_hist_length: 5
  });

  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  // Poll API Health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/health`);
        if (res.ok) {
          const data = await res.json();
          setApiOnline(data.status === 'healthy');
        } else {
          setApiOnline(false);
        }
      } catch (err) {
        setApiOnline(false);
      }
    };
    checkHealth();
    const timer = setInterval(checkHealth, 5000);
    return () => clearInterval(timer);
  }, []);

  // Compute Live Debt-to-Income
  const currentDti = Math.min(
    Math.round((formData.loan_amnt / (formData.person_income || 1)) * 100),
    100
  );

  // Quick Preset Profiles
  const loadPreset = (type) => {
    if (type === 'prime') {
      setFormData({
        person_age: 34,
        person_income: 95000,
        person_home_ownership: 'OWN',
        person_emp_length: 8.0,
        loan_intent: 'HOMEIMPROVEMENT',
        loan_grade: 'A',
        loan_amnt: 8000,
        loan_int_rate: 7.5,
        cb_person_default_on_file: 'N',
        cb_person_cred_hist_length: 10
      });
    } else if (type === 'distressed') {
      setFormData({
        person_age: 22,
        person_income: 22000,
        person_home_ownership: 'RENT',
        person_emp_length: 1.0,
        loan_intent: 'DEBTCONSOLIDATION',
        loan_grade: 'F',
        loan_amnt: 28000,
        loan_int_rate: 21.5,
        cb_person_default_on_file: 'Y',
        cb_person_cred_hist_length: 2
      });
    } else if (type === 'borderline') {
      setFormData({
        person_age: 27,
        person_income: 48000,
        person_home_ownership: 'MORTGAGE',
        person_emp_length: 3.0,
        loan_intent: 'EDUCATION',
        loan_grade: 'C',
        loan_amnt: 14000,
        loan_int_rate: 13.5,
        cb_person_default_on_file: 'N',
        cb_person_cred_hist_length: 4
      });
    }
  };

  // Submit Application
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg(null);

    try {
      const response = await fetch(`${API_BASE}/api/v1/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail?.[0]?.msg || errorData.detail || 'Prediction failed');
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setErrorMsg(err.message || 'Unable to connect to FastAPI prediction server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-white/10 bg-slate-950/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 to-sky-400 flex items-center justify-center text-white shadow-lg shadow-sky-500/20">
              <Building2 className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-heading font-extrabold text-lg text-white">FinRisk AI</h1>
                <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20">
                  ML Underwriting
                </span>
              </div>
              <p className="text-xs text-slate-400">Credit Default Risk Assessment System</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-xs text-slate-300">
              <span className={`w-2 h-2 rounded-full ${apiOnline ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
              <span>{apiOnline ? 'Model Online (FastAPI)' : 'API Disconnected'}</span>
            </div>
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="text-xs font-semibold text-sky-400 hover:text-sky-300 transition-colors"
            >
              API Docs →
            </a>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Preset Action Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 mb-8 glass-panel p-4 rounded-2xl border border-white/5">
          <div className="flex items-center gap-2 text-sm text-slate-300">
            <Sparkles className="w-4 h-4 text-sky-400" />
            <span className="font-medium">Quick Demo Profiles:</span>
          </div>
          <div className="flex flex-wrap gap-2.5">
            <button
              onClick={() => loadPreset('prime')}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/20 transition-all flex items-center gap-1.5"
            >
              🟢 Prime Profile (Low Risk)
            </button>
            <button
              onClick={() => loadPreset('borderline')}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/10 text-amber-300 border border-amber-500/30 hover:bg-amber-500/20 transition-all flex items-center gap-1.5"
            >
              🟡 Moderate Profile (Medium Risk)
            </button>
            <button
              onClick={() => loadPreset('distressed')}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-500/10 text-rose-300 border border-rose-500/30 hover:bg-rose-500/20 transition-all flex items-center gap-1.5"
            >
              🔴 Distressed Profile (High Risk)
            </button>
          </div>
        </div>

        {/* Dashboard 2-Column Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Form Simulator */}
          <div className="lg:col-span-7 space-y-6">
            <div className="glass-panel p-7 rounded-3xl border border-white/10 shadow-2xl">
              <h2 className="font-heading font-bold text-xl text-white mb-1">Applicant Parameters</h2>
              <p className="text-xs text-slate-400 mb-6">Modify financial and credit inputs to evaluate real-time default probability.</p>

              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Sliders: Income & Loan Amount */}
                <div className="space-y-4">
                  {/* Income Slider */}
                  <div className="bg-slate-900/60 p-4 rounded-xl border border-white/5">
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-xs font-semibold text-slate-300">Annual Gross Income</label>
                      <span className="font-heading font-bold text-sky-400 text-base">
                        ${Number(formData.person_income).toLocaleString()}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="5000"
                      max="300000"
                      step="1000"
                      value={formData.person_income}
                      onChange={(e) => setFormData({ ...formData, person_income: Number(e.target.value) })}
                      className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>

                  {/* Loan Amount Slider */}
                  <div className="bg-slate-900/60 p-4 rounded-xl border border-white/5">
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-xs font-semibold text-slate-300">Requested Loan Principal</label>
                      <span className="font-heading font-bold text-sky-400 text-base">
                        ${Number(formData.loan_amnt).toLocaleString()}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="1000"
                      max="100000"
                      step="500"
                      value={formData.loan_amnt}
                      onChange={(e) => setFormData({ ...formData, loan_amnt: Number(e.target.value) })}
                      className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>

                  {/* Interest Rate Slider */}
                  <div className="bg-slate-900/60 p-4 rounded-xl border border-white/5">
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-xs font-semibold text-slate-300">Loan Interest Rate</label>
                      <span className="font-heading font-bold text-sky-400 text-base">
                        {formData.loan_int_rate}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="3.0"
                      max="28.0"
                      step="0.1"
                      value={formData.loan_int_rate}
                      onChange={(e) => setFormData({ ...formData, loan_int_rate: Number(e.target.value) })}
                      className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                </div>

                {/* Live DTI Gauge Preview */}
                <div className="p-4 rounded-xl bg-slate-900/40 border border-white/5">
                  <div className="flex justify-between items-center text-xs mb-2">
                    <span className="text-slate-400">Live Debt-to-Income Burden:</span>
                    <span className={`font-bold ${currentDti > 35 ? 'text-rose-400' : 'text-emerald-400'}`}>
                      {currentDti}% {currentDti > 35 ? '(High Risk Zone)' : '(Safe Zone)'}
                    </span>
                  </div>
                  <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-300 ${
                        currentDti > 35 ? 'bg-rose-500' : currentDti > 20 ? 'bg-amber-400' : 'bg-emerald-400'
                      }`}
                      style={{ width: `${currentDti}%` }}
                    />
                  </div>
                </div>

                {/* Detailed Inputs Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Age */}
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1.5">Applicant Age</label>
                    <input
                      type="number"
                      min="18"
                      max="85"
                      value={formData.person_age}
                      onChange={(e) => setFormData({ ...formData, person_age: Number(e.target.value) })}
                      className="w-full px-3.5 py-2.5 rounded-xl bg-slate-900 border border-white/10 text-white text-sm focus:border-sky-400 focus:outline-none"
                    />
                  </div>

                  {/* Employment Length */}
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1.5">Employment Length (Years)</label>
                    <input
                      type="number"
                      min="0"
                      max="50"
                      step="0.5"
                      value={formData.person_emp_length}
                      onChange={(e) => setFormData({ ...formData, person_emp_length: Number(e.target.value) })}
                      className="w-full px-3.5 py-2.5 rounded-xl bg-slate-900 border border-white/10 text-white text-sm focus:border-sky-400 focus:outline-none"
                    />
                  </div>

                  {/* Home Ownership */}
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1.5">Home Ownership</label>
                    <select
                      value={formData.person_home_ownership}
                      onChange={(e) => setFormData({ ...formData, person_home_ownership: e.target.value })}
                      className="w-full px-3.5 py-2.5 rounded-xl bg-slate-900 border border-white/10 text-white text-sm focus:border-sky-400 focus:outline-none"
                    >
                      <option value="RENT">Rent</option>
                      <option value="OWN">Own</option>
                      <option value="MORTGAGE">Mortgage</option>
                      <option value="OTHER">Other</option>
                    </select>
                  </div>

                  {/* Loan Intent */}
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1.5">Loan Purpose / Intent</label>
                    <select
                      value={formData.loan_intent}
                      onChange={(e) => setFormData({ ...formData, loan_intent: e.target.value })}
                      className="w-full px-3.5 py-2.5 rounded-xl bg-slate-900 border border-white/10 text-white text-sm focus:border-sky-400 focus:outline-none"
                    >
                      <option value="PERSONAL">Personal</option>
                      <option value="EDUCATION">Education</option>
                      <option value="MEDICAL">Medical</option>
                      <option value="VENTURE">Venture</option>
                      <option value="HOMEIMPROVEMENT">Home Improvement</option>
                      <option value="DEBTCONSOLIDATION">Debt Consolidation</option>
                    </select>
                  </div>

                  {/* Loan Grade */}
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1.5">Credit Risk Grade</label>
                    <select
                      value={formData.loan_grade}
                      onChange={(e) => setFormData({ ...formData, loan_grade: e.target.value })}
                      className="w-full px-3.5 py-2.5 rounded-xl bg-slate-900 border border-white/10 text-white text-sm focus:border-sky-400 focus:outline-none font-bold text-sky-400"
                    >
                      <option value="A">Grade A (Prime - Lowest Risk)</option>
                      <option value="B">Grade B</option>
                      <option value="C">Grade C</option>
                      <option value="D">Grade D</option>
                      <option value="E">Grade E</option>
                      <option value="F">Grade F</option>
                      <option value="G">Grade G (Subprime - Extreme Risk)</option>
                    </select>
                  </div>

                  {/* Historical Default On File */}
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1.5">Prior Bureau Default Flag</label>
                    <select
                      value={formData.cb_person_default_on_file}
                      onChange={(e) => setFormData({ ...formData, cb_person_default_on_file: e.target.value })}
                      className="w-full px-3.5 py-2.5 rounded-xl bg-slate-900 border border-white/10 text-white text-sm focus:border-sky-400 focus:outline-none"
                    >
                      <option value="N">No (Clean Credit Record)</option>
                      <option value="Y">Yes (Previous Default on File)</option>
                    </select>
                  </div>
                </div>

                {/* Submit Action */}
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full py-4 px-6 rounded-2xl bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-400 hover:to-blue-500 text-white font-heading font-bold text-base shadow-xl shadow-sky-500/25 transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-5 h-5 animate-spin" />
                      Evaluating Risk Parameters...
                    </>
                  ) : (
                    <>
                      <Send className="w-5 h-5" />
                      Run ML Underwriting Assessment
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>

          {/* Right Column: Assessment Results */}
          <div className="lg:col-span-5 space-y-6">
            {errorMsg && (
              <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-start gap-3">
                <ShieldAlert className="w-5 h-5 flex-shrink-0 text-rose-400 mt-0.5" />
                <div>
                  <p className="font-bold">Assessment Failed</p>
                  <p className="text-xs text-rose-400/80 mt-1">{errorMsg}</p>
                </div>
              </div>
            )}

            {prediction ? (
              <div className="space-y-6 animate-fadeIn">
                {/* Circular Score Gauge */}
                <RiskGauge
                  approvalProbability={prediction.approval_probability}
                  defaultProbability={prediction.default_probability}
                  riskTier={prediction.risk_tier}
                  loanStatus={prediction.loan_status}
                />

                {/* Mini Metrics Grid */}
                <div className="grid grid-cols-2 gap-3.5">
                  <MetricCard
                    label="Default Risk"
                    value={`${(prediction.default_probability * 100).toFixed(1)}%`}
                    subtext="XGBoost Estimated"
                    icon={Percent}
                    color="text-rose-400"
                  />
                  <MetricCard
                    label="Model Confidence"
                    value={`${prediction.confidence_score}%`}
                    subtext="Calibrated Probability"
                    icon={Activity}
                    color="text-sky-400"
                  />
                  <MetricCard
                    label="Debt Burden"
                    value={`${prediction.debt_to_income_ratio}%`}
                    subtext="Loan-to-Income"
                    icon={TrendingUp}
                    color="text-amber-400"
                  />
                  <MetricCard
                    label="Decision Status"
                    value={prediction.loan_status}
                    subtext={prediction.risk_tier}
                    icon={UserCheck}
                    color={prediction.loan_status === 'Approved' ? 'text-emerald-400' : 'text-rose-400'}
                  />
                </div>

                {/* Explainable Decision Factors */}
                <RiskFactorsList
                  factors={prediction.risk_factors}
                  isDeclined={prediction.loan_status === 'Declined'}
                />
              </div>
            ) : (
              <div className="glass-panel p-12 rounded-3xl border border-white/5 text-center flex flex-col items-center justify-center min-h-[460px]">
                <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center text-slate-500 mb-4">
                  <Activity className="w-8 h-8 opacity-60" />
                </div>
                <h3 className="font-heading font-bold text-lg text-white mb-2">No Active Assessment</h3>
                <p className="text-xs text-slate-400 max-w-xs leading-relaxed">
                  Adjust the applicant parameters or select a quick demo profile on the left to trigger real-time ML risk scoring.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
