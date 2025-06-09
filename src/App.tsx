import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import axios from "axios";
import LaptopCard from "@/components/LaptopCard";
import { motion, AnimatePresence } from "framer-motion";
import { SpinnerGap, Sparkle, ArrowCounterClockwise } from "phosphor-react";

export type LaptopRecommendation = {
  Brand: string;
  Model: string;
  CPU: string;
  GPU: string;
  Price: number | string;
  Score?: number;
  Category?: string;
};

const App: React.FC = () => {
  const [stage, setStage] = useState<"init" | "welcome" | "loading" | "form" | "result">("init");
  const [teks, setTeks] = useState("");
  const [rekomendasi, setRekomendasi] = useState<LaptopRecommendation[]>([]);
  const [typewriterText, setTypewriterText] = useState("");
  const fullTextRef = useRef("Selamat Datang di LapPick ~");
  const idxRef = useRef(0);

  useEffect(() => {
    if (stage === "welcome") {
      setTypewriterText("");
      idxRef.current = 0;

      const interval = setInterval(() => {
        const current = idxRef.current;
        const full = fullTextRef.current;

        if (current < full.length) {
          setTypewriterText((prev) => prev + full[current]);
          idxRef.current++;
        } else {
          clearInterval(interval);
          setTimeout(() => {
            setStage("loading");
            setTimeout(() => setStage("form"), 1000);
          }, 1000);
        }
      }, 60);

      return () => clearInterval(interval);
    }
  }, [stage]);

  const handleStart = () => setStage("welcome");

  const handleSubmit = async () => {
    if (!teks.trim()) return toast.error("Input kosong");
    try {
      const res = await axios.get<{ rekomendasi: LaptopRecommendation[] }>(
        `http://localhost:5000/rekomendasi?teks=${encodeURIComponent(teks)}`
      );
      setRekomendasi(res.data.rekomendasi ?? []);
      setStage("result");
    } catch {
      toast.error("Gagal mengambil data.");
    }
  };

  const resetAll = () => {
    setTeks("");
    setRekomendasi([]);
    setStage("init");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#1e1e3f] via-[#302b63] to-[#5c258d] text-white relative flex items-center justify-center overflow-hidden">

      {/* INIT */}
      <AnimatePresence>
        {stage === "init" && (
          <motion.div
            key="init"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.3 }}
              className="bg-gradient-to-br from-[#2a2a5e]/90 to-[#3c2e6c]/90 rounded-2xl p-10 w-full max-w-lg shadow-xl shadow-purple-500/10 border border-white/10 text-center"
            >
              <motion.div
                className="text-5xl mb-4 animate-wiggle origin-center"
                initial={{ rotate: -10 }}
                animate={{ rotate: [0, -10, 10, -5, 5, 0] }}
                transition={{ duration: 1.2 }}
              >
                👋
              </motion.div>
              <h1 className="text-3xl sm:text-4xl font-bold leading-tight mb-2">
                Siap mulai cari laptop terbaik?
              </h1>
              <p className="text-sm text-zinc-400 mb-6">Klik tombol di bawah ini untuk memulai</p>
              <Button
                onClick={handleStart}
                className="w-full bg-purple-600 hover:bg-purple-700 transition-all duration-300 shadow-md hover:shadow-purple-500/40"
              >
                Mulai
              </Button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* POPUP STAGES */}
      <AnimatePresence>
        {stage !== "init" && stage !== "result" && (
          <motion.div
            key="popup"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          >
            <motion.div
              key={stage}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.3 }}
              className="bg-gradient-to-br from-zinc-900 to-zinc-800 text-white rounded-2xl p-10 w-full max-w-lg space-y-6 shadow-xl shadow-purple-500/10 border border-zinc-700 text-center"
            >
              {stage === "welcome" && (
                <>
                  <Sparkle className="mx-auto text-purple-400 w-10 h-10 animate-pulse" weight="duotone" />
                  <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight min-h-[3rem] leading-snug">
                    {typewriterText}
                    <span className="animate-blink">|</span>
                  </h1>
                  {typewriterText === fullTextRef.current && (
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.5 }}
                      className="text-sm text-zinc-400"
                    >
                      Rekomendasi laptop berdasarkan kebutuhanmu
                    </motion.p>
                  )}
                </>
              )}

              {stage === "loading" && (
                <div className="flex flex-col items-center gap-3">
                  <SpinnerGap className="w-8 h-8 animate-spin text-purple-400" />
                  <p className="text-sm text-zinc-400">Menyiapkan input form...</p>
                </div>
              )}

              {stage === "form" && (
                <>
                  <h2 className="text-xl font-bold text-center mb-4">Apa kebutuhan laptopmu?</h2>
                  <Input
                    placeholder="Contoh: laptop untuk desain grafis 10 juta"
                    value={teks}
                    onChange={(e) => setTeks(e.target.value)}
                    className="bg-zinc-800 border border-zinc-600 hover:border-purple-400 focus:ring-2 focus:ring-purple-500 text-white placeholder:text-zinc-400/70 transition duration-200 ease-in-out"
                  />
                  <Button
                    onClick={handleSubmit}
                    className="w-full bg-purple-600 hover:bg-purple-700 transition duration-200"
                  >
                    Cari
                  </Button>
                </>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* RESULT */}
      <AnimatePresence>
        {stage === "result" && (
          <motion.div
            key="result"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
            className="p-6 space-y-6 w-full max-w-6xl mx-auto"
          >
            <div className="flex justify-between items-center mb-4">
              <h1 className="text-3xl font-bold text-purple-400">Rekomendasi Laptop untukmu</h1>
              <Button variant="outline" onClick={resetAll} className="gap-2">
                <ArrowCounterClockwise size={20} /> Ulangi
              </Button>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {rekomendasi.map((item, idx) => (
                <LaptopCard key={idx} data={item} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Floating Glow Animation */}
      <motion.div
        animate={{ x: [0, 40, -40, 0], y: [0, -20, 20, 0] }}
        transition={{ duration: 20, repeat: Infinity }}
        className="absolute -top-40 -left-40 w-96 h-96 bg-purple-500 rounded-full blur-3xl opacity-10"
      />
    </div>
  );
};

export default App;
