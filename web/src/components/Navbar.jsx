import React from 'react';
import { Activity, LayoutDashboard, Home as HomeIcon } from 'lucide-react';

const Navbar = ({ onNavigate, currentPage }) => {
    return (
        <nav className="fixed top-0 left-0 right-0 h-[72px] glass z-50 flex items-center justify-between px-8 border-b border-white/10">
            <div className="flex items-center gap-3 cursor-pointer" onClick={() => onNavigate('home')}>
                <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/40">
                    <Activity className="text-white w-6 h-6" />
                </div>
                <span className="text-2xl font-bold tracking-tight font-display text-gradient">deTACTer</span>
            </div>

            <div className="flex items-center gap-12">
                <button
                    onClick={() => onNavigate('home')}
                    className={`flex items-center gap-4 px-10 py-3 rounded-2xl transition-all font-bold tracking-widest text-sm ${currentPage === 'home'
                        ? 'bg-indigo-600 text-white border border-indigo-400 shadow-[0_0_25px_rgba(99,102,241,0.5)] scale-110'
                        : 'glass-button text-slate-400 hover:text-white'
                        }`}
                >
                    <HomeIcon size={20} />
                    <span>HOME</span>
                </button>
                <button
                    onClick={() => onNavigate('analysis')}
                    className={`flex items-center gap-4 px-10 py-3 rounded-2xl transition-all font-bold tracking-widest text-sm ${currentPage === 'analysis'
                        ? 'bg-indigo-600 text-white border border-indigo-400 shadow-[0_0_25px_rgba(99,102,241,0.5)] scale-110'
                        : 'glass-button text-slate-400 hover:text-white'
                        }`}
                >
                    <LayoutDashboard size={20} />
                    <span>ANALYSIS</span>
                </button>
            </div>

            <div className="flex items-center gap-4">
                <div className="text-sm font-medium text-slate-400 bg-slate-800/50 px-3 py-1 rounded-full border border-white/5">
                    v5.7.0
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
