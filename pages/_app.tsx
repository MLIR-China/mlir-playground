import "../styles/globals.css";
import type { AppProps } from "next/app";
import { ChakraProvider } from "@chakra-ui/react";
import PlausibleProvider from "next-plausible";

function MyApp({ Component, pageProps }: AppProps) {
  const page = (
    <ChakraProvider>
      <Component {...pageProps} />
    </ChakraProvider>
  );

  if (process.env.productionDomain) {
    return (
      <PlausibleProvider domain={process.env.productionDomain}>
        {page}
      </PlausibleProvider>
    );
  }
  return page;
}

export default MyApp;
