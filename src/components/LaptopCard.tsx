import React from "react"

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

  return (
    <div className="bg-zinc-800 rounded-xl p-4 shadow-md space-y-1">
      <h2 className="text-xl font-semibold">{Brand} {Model}</h2>
      <p className="text-sm text-zinc-400">{CPU} | {GPU}</p>
      <p className="text-sm">Score: {Score ?? "—"} | Kategori: {Category ?? "—"}</p>
      <p className="text-lg font-bold text-lime-400">
        Rp {typeof Price === "number" ? Price.toLocaleString() : parseInt(Price).toLocaleString()}
      </p>
    </div>
  )
}

export default LaptopCard
