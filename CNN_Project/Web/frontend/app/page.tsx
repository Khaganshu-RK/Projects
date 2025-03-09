export default function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center py-2">
      <main className="flex w-full flex-1 flex-col items-center justify-center px-20 text-center">
        <h1 className="text-6xl font-bold">
          CNN Age Gender Ethnicity Classification Project{" "}
          <span className="text-blue-600">👁️</span>
        </h1>

        <p className="mt-3 text-2xl">
          Created by{" "}
          <a
            href="https://github.com/Khaganshu-RK"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline">
            Khaganshu R Khobragade
          </a>
        </p>
        {/* Click below button to start the prediction */}
        <p className="mt-3 text-2xl">
          <a
            href="form"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline">
            Click here to start the prediction
          </a>
        </p>
      </main>
    </div>
  );
}
