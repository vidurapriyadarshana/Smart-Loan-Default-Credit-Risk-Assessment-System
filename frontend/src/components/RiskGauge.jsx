import React from 'react';
import { ShieldCheck, AlertTriangle, ShieldX } from 'lucide-react';

export default function RiskGauge({ approvalProbability, defaultProbability, riskTier, loanStatus }) {
  // SVG circular gauge calculation: radius = 75, circumference = 2 * PI * 75 ≈ 471.24
  const radius = 75;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - approvalProbability);

  // Dynamic status styling
  let strokeColor = '#10b981'; // Green (Approved)
  let statusBadgeColor = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
  let StatusIcon = ShieldCheck;

  if (riskTier === 'Medium Risk') {
    strokeColor = '#f59e0b'; // Amber (Review)
    statusBadgeColor = 'bg-amber-500/10 text-amber-400 border-amber-500/30';
    StatusIcon = AlertTriangle;
  } else if (riskTier === 'High Risk') {
    strokeColor = '#ef4444'; // Red (Declined)
    statusBadgeColor = 'bg-rose-500/10 text-rose-400 border-rose-500/30';
    StatusIcon = ShieldX;
  }

  const percentage = Math.round(approvalProbability * 100);

  return (
    <div className="flex flex-col items-center justify-center p-6 glass-panel rounded-2xl border border-white/5 relative overflow-hidden">
      <div className="relative flex items-center justify-center">
        <svg className="w-52 h-52 transform -rotate-90" viewBox="0 0 180 180">
          {/* Track Background */}
          <circle
            cx="90"
            cy="90"
            r={radius}
            className="stroke-slate-800"
            strokeWidth="12"
            fill="transparent"
          />
          {/* Animated Progress Arc */}
          <circle
            cx="90"
            cy="90"
            r={radius}
            stroke={strokeColor}
            strokeWidth="12"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            fill="transparent"
            style={{
              transition: 'stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.4s ease'
            }}
          />
        </svg>

        {/* Center Readout */}
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          <span className="text-4xl font-extrabold tracking-tight font-heading text-white">
            {percentage}%
          </span>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mt-1">
            Approval Odds
          </span>
        </div>
      </div>

      {/* Decision Pill */}
      <div className={`mt-6 inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-bold border ${statusBadgeColor}`}>
        <StatusIcon className="w-4 h-4" />
        <span>{loanStatus.toUpperCase()} — {riskTier.toUpperCase()}</span>
      </div>
    </div>
  );
}
