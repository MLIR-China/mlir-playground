import type { NextPage } from 'next'
import Head from 'next/head'
import Image from 'next/image'
import Editor, { OnMount } from '@monaco-editor/react'
import styles from '../styles/Home.module.css'

import {
  Button,
  Grid,
  GridItem,
  Heading,
  VStack,
  Box,
  HStack,
  Textarea
} from '@chakra-ui/react'

import { useRef, useState } from 'react'
import WasmCompiler from '../components/WasmCompiler/index.js'

const Playground: NextPage = () => {
  const defaultCode =
`#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Custom optimizer driver\\n", registry));
}
`
  const defaultMLIRInput =
`module  {
  func @main() -> i32 {
    %0 = constant 42 : i32
    return %0 : i32
  }
}
`
  const defaultMLIROutput = ""

  const cppEditor : React.MutableRefObject<any> = useRef();
  const inputEditor : React.MutableRefObject<any> = useRef();
  const outputEditor : React.MutableRefObject<any> = useRef();
  const [logValue, setLogValue] = useState('');

  const onCppEditorMount : OnMount = (editor, _) => {
    cppEditor.current = editor;
  }

  const onInputViewerMount : OnMount = (editor, _) => {
    inputEditor.current = editor;
  }

  const onOutputViewerMount : OnMount = (editor, _) => {
    outputEditor.current = editor;
  }

  const onRunButtonClick = () => {
    let cpp_source = cppEditor.current.getValue();
    let input_mlir = inputEditor.current.getValue();
    let printer = (text: string) => {
      setLogValue(currValue => currValue + text + "\n");
    };
    WasmCompiler()
      .compileAndRun(cpp_source, input_mlir, ["--my-pass"], printer)
      .then((output: string) => { outputEditor.current.setValue(output); }, printer);
  }

  const monacoOptions = {
    selectOnLineNumbers: true,
    quickSuggestions: true,
    minimap: {
      enabled: false,
    },
    scrollBeyondLastLine: false,
    automaticLayout: true,
  }

  const codeEditor = (
    <Editor
      key="cppEditor"
      height="100%"
      defaultLanguage="cpp"
      defaultValue={defaultCode}
      onMount={onCppEditorMount}
      options={monacoOptions}
    />
  )

  const inputMLIRViewer = (
    <Editor
      key="inputEditor"
      height="100%"
      defaultLanguage="cpp"
      defaultValue={defaultMLIRInput}
      onMount={onInputViewerMount}
      options={monacoOptions}
    />
  )

  const outputMLIRViewer = (
    <Editor
      key="outputEditor"
      height="100%"
      defaultLanguage="cpp"
      defaultValue={defaultMLIROutput}
      onMount={onOutputViewerMount}
      options={{...monacoOptions, readOnly: true}}
    />
  )

  return (
    <div className={styles.container}>
      <Head>
        <title>MLIR Playground</title>
        <meta name="description" content="Playing with MLIR right in the browser." />
        <link rel="icon" href="/mlir.png" />
      </Head>
      <main className={styles.main_playground}>
        <Grid
          templateRows="repeat(3, 1fr)"
          templateColumns="repeat(2, 1fr)"
          columnGap={2}
        >
          <GridItem rowSpan={3} colSpan={1} h="800">
            <VStack spacing={4} align="left" h="100%">
              <HStack>
                <Heading>Editor</Heading>
                <Button
                  isLoading={!inputEditor || !outputEditor}
                  mt="8"
                  as="a"
                  size="lg"
                  colorScheme="blue"
                  fontWeight="bold"
                  onClick={onRunButtonClick}
                >
                  Run
                </Button>
              </HStack>
              <Box borderWidth="2px" height="100%">
                {codeEditor}
              </Box>
            </VStack>
          </GridItem>
          <GridItem rowSpan={1} colSpan={1} h="200" marginTop={1}>
            <Heading>Input</Heading>
            <Box borderWidth="2px" height="100%">
              {inputMLIRViewer}
            </Box>
          </GridItem>
          <GridItem rowSpan={1} colSpan={1} h="200" marginTop={1}>
            <Heading>Output</Heading>
            <Box borderWidth="2px" height="100%">
              {outputMLIRViewer}
            </Box>
          </GridItem>
          <GridItem rowSpan={1} colSpan={1} marginTop={1}>
            <Heading>Logs</Heading>
            <Textarea borderWidth="2px" height="100%" bg="gray.800" value={logValue} readOnly color="white" fontFamily="Consolas, 'Courier New', monospace"></Textarea>
          </GridItem>
        </Grid>
      </main>
    </div>
  )
}

export default Playground
