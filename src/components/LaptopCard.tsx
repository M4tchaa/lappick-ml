import React from "react"
import { Cpu} from "phosphor-react"

export type LaptopRecommendation = {
  Brand: string
  Model: string
  CPU: string
  GPU: string
  Price: number | string
  Score?: number
  Category?: string
}

type Props = {
  data: LaptopRecommendation
}

const LaptopCard: React.FC<Props> = ({ data }) => {
  const { Brand, Model, CPU, GPU, Price, Score, Category } = data
  const priceValue = typeof Price === "number" ? Price : parseInt(Price)

  return (
    <div className="bg-zinc-800 rounded-xl p-5 shadow-lg border border-zinc-700 hover:shadow-purple-500/20 transition-all duration-300 space-y-3">
      {/* Title */}
      <h2 className="text-lg font-bold text-white leading-snug">
        {Brand} <span className="text-purple-400">{Model}</span>
      </h2>

      {/* Category */}
      {Category && (
        <span className="inline-block text-xs bg-purple-900/50 text-purple-300 px-2 py-1 rounded-full uppercase tracking-wide">
          {Category}
        </span>
      )}

      {/* CPU & GPU */}
      <div className="text-sm text-zinc-300 flex items-center gap-3">
        <div className="flex items-center gap-1">
          <Cpu size={16} /> <span>{CPU}</span>
        </div>
        <div className="flex items-center gap-1">
          <span>{GPU}</span>
        </div>
      </div>

      {/* Score Bar */}
      {typeof Score === "number" && (
        <div className="text-xs text-zinc-400">
          Score:
          <div className="w-full bg-zinc-700 h-2 rounded mt-1 overflow-hidden">
            <div
              className="bg-lime-400 h-full"
              style={{ width: `${Math.min(Score / 300, 1) * 100}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* Price */}
      <p className="text-lg font-semibold text-lime-400">
        Rp {priceValue.toLocaleString("id-ID")}
      </p>
    </div>
  )
}

export default LaptopCard