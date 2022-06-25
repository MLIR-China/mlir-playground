import React, { useRef, useState } from 'react'
import type { NextPage } from 'next'
import Head from 'next/head'

import {
  Box,
  Button,
  Flex,
  Grid,
  GridItem,
  Heading,
  HStack,
  Input,
  InputGroup,
  InputLeftAddon,
  InputRightAddon,
  Select,
  Spacer,
  Text,
  Textarea,
  VStack
} from '@chakra-ui/react'
import Editor, { OnMount } from '@monaco-editor/react'

import Toy from '../components/Toy/index.js'
import WasmCompiler from '../components/WasmCompiler/index.js'
import styles from '../styles/Home.module.css'

const Home: NextPage = () => {
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

  const defaultToyInput =
`def main() {
  print([[1, 2], [3, 4]]);
}
`

  const defaultMLIROutput = ""

  const monospaceFontFamily = "Consolas, 'Courier New', monospace";

  enum ProgramSelectionOption {
    MlirOpt = "Custom mlir-opt",
    Toy1 = "Toy Chapter 1",
    Toy2 = "Toy Chapter 2",
    Toy3 = "Toy Chapter 3",
    Toy4 = "Toy Chapter 4",
    Toy5 = "Toy Chapter 5",
    Toy6 = "Toy Chapter 6",
    Toy7 = "Toy Chapter 7"
  }

  abstract class ProgramProperties {
    abstract getInputFileName(): string;
    abstract getOutputFileName(): string;
    abstract getDefaultInputFile(): string;
    abstract getDefaultAdditionalRunArgs(): string;
    abstract getRunArgsLeftAddon(): string;
    abstract getRunArgsRightAddon(): string;
  }

  class MlirOptProperties extends ProgramProperties {
    getInputFileName(): string { return "input.mlir"; }
    getOutputFileName(): string { return "output.mlir"; }
    getDefaultInputFile(): string { return defaultMLIRInput; }
    getDefaultAdditionalRunArgs(): string { return "--convert-std-to-llvm"; }
    getRunArgsLeftAddon(): string { return "mlir-opt"; }
    getRunArgsRightAddon(): string { return "input.mlir -o output.mlir"; }
  }

  class ToyChapterProperties extends ProgramProperties {
    chapterNumber: number;
    constructor(chapterNumber: number) {
      super();
      this.chapterNumber = chapterNumber;
    }
    getInputFileName(): string { return "input.toy"; }
    getOutputFileName(): string { return "output.mlir"; }
    getDefaultInputFile(): string { return defaultToyInput; }
    getDefaultAdditionalRunArgs(): string { return "--emit=mlir"; }
    getRunArgsLeftAddon(): string { return "toy input.toy"; }
    getRunArgsRightAddon(): string { return ""; }
  }

  const getProgramProperties = (selection: ProgramSelectionOption) => {
    if (selection == ProgramSelectionOption.MlirOpt) {
      return new MlirOptProperties();
    } else {
      let chapterNumber = parseInt(selection.slice(-1));
      return new ToyChapterProperties(chapterNumber);
    }
  }

  // state
  const [allEditorsMounted, setAllEditorsMounted] = useState(false);
  const cppEditor : React.MutableRefObject<any> = useRef(null);
  const inputEditor : React.MutableRefObject<any> = useRef(null);
  const outputEditor : React.MutableRefObject<any> = useRef(null);
  const [logValue, setLogValue] = useState('');
  
  const [programSelection, setProgramSelection] = React.useState(ProgramSelectionOption.MlirOpt);

  const [runArgsLeftAddon, setRunArgsLeftAddon] = useState("mlir-opt");
  const [runArgsRightAddon, setRunArgsRightAddon] = useState("input.mlir -o output.mlir");
  const [additionalRunArgs, setAdditionalRunArgs] = useState("--convert-std-to-llvm");
  const [inputEditorFileName, setInputEditorFileName] = useState("input.mlir");
  const [outputEditorFileName, setOutputEditorFileName] = useState("output.mlir");

  const wasmCompiler : React.MutableRefObject<any> = useRef(null);
  const [compilerState, setCompilerState] = useState("");

  const getWasmCompiler = () => {
    if (!wasmCompiler.current) {
      wasmCompiler.current = WasmCompiler();
    }
    return wasmCompiler.current;
  };

  const onEditorMounted = (editorRef: React.MutableRefObject<any>) : OnMount => {
    return (editor, _) => {
      editorRef.current = editor;
      if (cppEditor.current && inputEditor.current && outputEditor.current) {
        // All editors mounted.
        setAllEditorsMounted(true);
      }
    }
  }

  const onProgramSelectionChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setProgramSelection(event.target.value as ProgramSelectionOption);
    const selectedOption = ProgramSelectionOption[event.target.value as keyof typeof ProgramSelectionOption];
    cppEditor.current.updateOptions({
      readOnly: selectedOption != ProgramSelectionOption.MlirOpt
    });
    const props = getProgramProperties(selectedOption);
    setRunArgsLeftAddon(props.getRunArgsLeftAddon());
    setRunArgsRightAddon(props.getRunArgsRightAddon());
    setAdditionalRunArgs(props.getDefaultAdditionalRunArgs());
    setInputEditorFileName(props.getInputFileName());
    setOutputEditorFileName(props.getOutputFileName());
    inputEditor.current.setValue(props.getDefaultInputFile());
    outputEditor.current.setValue("");
  }

  const onRunButtonClick = () => {
    let input_mlir = inputEditor.current.getValue();
    let printer = (text: string) => {
      setLogValue(currValue => currValue + text + "\n");
    };

    if (programSelection == ProgramSelectionOption.MlirOpt) {
      let cpp_source = cppEditor.current.getValue();
      setCompilerState("Compiling...");
      getWasmCompiler()
        .compileAndRun(cpp_source, input_mlir, additionalRunArgs.split(/\s+/), printer)
        .finally(() => { setCompilerState(""); })
        .then((output: string) => { outputEditor.current.setValue(output); }, printer);
    } else {
      let chapterIndex = parseInt(programSelection.slice(-1));
      setCompilerState("Running...");
      Toy.runChapter(chapterIndex, input_mlir, additionalRunArgs.split(/\s+/), printer)
        .finally(() => { setCompilerState(""); })
        .then((output: string) => { outputEditor.current.setValue(output); }, printer)
    }
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
          templateRows="repeat(4, 1fr)"
          templateColumns="repeat(2, 1fr)"
          columnGap={4}
          rowGap={2}
        >
          <GridItem rowSpan={4} colSpan={1}>
            <VStack spacing={4} align="left">
              <HStack>
                <Heading>MLIR Playground</Heading>
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
              <Box>
                <Select value={programSelection} onChange={onProgramSelectionChange} disabled={!allEditorsMounted}>
                  {
                    Object.keys(ProgramSelectionOption).map((key, i) => {
                      return <option value={key} key={i}>{ProgramSelectionOption[key as keyof typeof ProgramSelectionOption]}</option>;
                    })
                  }
                </Select>
              </Box>
              <HStack>
                <Text>Arguments</Text>
                <InputGroup fontFamily={monospaceFontFamily}>
                  <InputLeftAddon>{runArgsLeftAddon}</InputLeftAddon>
                  <Input
                    value={additionalRunArgs}
                    onChange={(event) => setAdditionalRunArgs(event.target.value)}></Input>
                  <InputRightAddon>{runArgsRightAddon}</InputRightAddon>
                </InputGroup>
              </HStack>
              <Box>
                <Flex align="end">
                  <Heading>Editor</Heading>
                  <Spacer />
                  <Text fontFamily={monospaceFontFamily}>mlir-opt.cpp</Text>
                </Flex>
                <Box borderWidth="2px" h="800">
                  {codeEditor}
                </Box>
              </Box>
            </VStack>
          </GridItem>
          <GridItem rowSpan={1} colSpan={1}>
            <Flex align="end">
              <Heading>Input</Heading>
              <Spacer />
              <Text fontFamily={monospaceFontFamily}>{inputEditorFileName}</Text>
            </Flex>
            <Box borderWidth="2px" h="200">
              {inputMLIRViewer}
            </Box>
          </GridItem>
          <GridItem rowSpan={1} colSpan={1}>
            <Flex align="end">
              <Heading>Output</Heading>
              <Spacer />
              <Text fontFamily={monospaceFontFamily}>{outputEditorFileName}</Text>
            </Flex>
            <Box borderWidth="2px" h="200">
              {outputMLIRViewer}
            </Box>
          </GridItem>
          <GridItem rowSpan={2} colSpan={1}>
            <Heading>Logs</Heading>
            <Textarea borderWidth="2px" height="100%" bg="gray.800" value={logValue} readOnly color="white" fontFamily={monospaceFontFamily}></Textarea>
          </GridItem>
        </Grid>
      </main>
    </div>
  )
}

export default Home
