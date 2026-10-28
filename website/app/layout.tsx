import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Financial Distress, Mapped | FME Research",
  description:
    "A reader's guide to a fiscal-year study of financial peer graphs and next-year bankruptcy prediction.",
  openGraph: {
    title: "Financial Distress, Mapped",
    description:
      "What changes when a temporal graph model learns from financial peers? Explore the data, model, evaluation, and measured results.",
    type: "article",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
