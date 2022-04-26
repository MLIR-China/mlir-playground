import React, { useRef, useState } from 'react'
import type { NextPage } from 'next'
import Head from 'next/head'
import Image from 'next/image'

import {
  Button,
  Grid,
  GridItem,
  Heading,
  VStack,
  Box,
  HStack,
  Text,
  Textarea
} from '@chakra-ui/react'
import Editor, { OnMount } from '@monaco-editor/react'

import WasmCompiler from '../components/WasmCompiler/index.js'
import styles from '../styles/Home.module.css'

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

  // state
  const [compilerState, setCompilerState] = useState("");
  const [allEditorsMounted, setAllEditorsMounted] = useState(false);
  const cppEditor : React.MutableRefObject<any> = useRef(null);
  const inputEditor : React.MutableRefObject<any> = useRef(null);
  const outputEditor : React.MutableRefObject<any> = useRef(null);
  const [logValue, setLogValue] = useState('');

  const onEditorMounted = (editorRef: React.MutableRefObject<any>) : OnMount => {
    return (editor, _) => {
      editorRef.current = editor;
      if (cppEditor.current && inputEditor.current && outputEditor.current) {
        // All editors mounted.
        setAllEditorsMounted(true);
      }
    }
  }

  const onRunButtonClick = () => {
    let cpp_source = cppEditor.current.getValue();
    let input_mlir = inputEditor.current.getValue();
    let printer = (text: string) => {
      setLogValue(currValue => currValue + text + "\n");
    };
    setCompilerState("Compiling...");
    WasmCompiler()
      .compileAndRun(cpp_source, input_mlir, ["--my-pass"], printer)
      .finally(() => { setCompilerState(""); })
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
      onMount={onEditorMounted(cppEditor)}
      options={monacoOptions}
    />
  )

  const inputMLIRViewer = (
    <Editor
      key="inputEditor"
      height="100%"
      defaultLanguage="cpp"
      defaultValue={defaultMLIRInput}
      onMount={onEditorMounted(inputEditor)}
      options={monacoOptions}
    />
  )

  const outputMLIRViewer = (
    <Editor
      key="outputEditor"
      height="100%"
      defaultLanguage="cpp"
      defaultValue={defaultMLIROutput}
      onMount={onEditorMounted(outputEditor)}
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
                  isLoading={!allEditorsMounted || compilerState !== ""}
                  mt="8"
                  as="a"
                  size="lg"
                  colorScheme="blue"
                  fontWeight="bold"
                  onClick={onRunButtonClick}
                >
                  Run
                </Button>
                <Text>{compilerState}</Text>
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
