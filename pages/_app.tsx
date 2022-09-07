import "../styles/globals.css";
import type { AppProps } from "next/app";
import { ChakraProvider } from "@chakra-ui/react";
import PlausibleProvider from "next-plausible";

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <PlausibleProvider domain="playground.mlir-china.org">
      <ChakraProvider>
        <Component {...pageProps} />
      </ChakraProvider>
    </PlausibleProvider>
  );
}

export default MyApp;
