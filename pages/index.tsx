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
import styles from '../styles/Home.module.css'

import Toy from '../components/Toy/index.js'
import WasmCompiler from '../components/WasmCompiler/index.js'

import { defaultFacility, getFacilityNames, getFacility } from '../components/Facilities/FacilitySelector'

const Home: NextPage = () => {
  const monospaceFontFamily = "Consolas, 'Courier New', monospace";

  // state
  const [allEditorsMounted, setAllEditorsMounted] = useState(false);
  const cppEditor : React.MutableRefObject<any> = useRef(null);
  const inputEditor : React.MutableRefObject<any> = useRef(null);
  const outputEditor : React.MutableRefObject<any> = useRef(null);
  const [logValue, setLogValue] = useState('');
  
  const [currentFacility, setCurrentFacility] = React.useState(defaultFacility);

  const [runArgsLeftAddon, setRunArgsLeftAddon] = useState("");
  const [runArgsRightAddon, setRunArgsRightAddon] = useState("");
  const [additionalRunArgs, setAdditionalRunArgs] = useState("");
  const [inputEditorFileName, setInputEditorFileName] = useState("");
  const [outputEditorFileName, setOutputEditorFileName] = useState("");

  const wasmCompiler : React.MutableRefObject<any> = useRef(null);
  const [compilerState, setCompilerState] = useState("");

  function getWasmCompiler() {
    if (!wasmCompiler.current) {
      wasmCompiler.current = WasmCompiler();
    }
    return wasmCompiler.current;
  };

  function setFacilitySelection(selection: string) {
    setCurrentFacility(selection);
    cppEditor.current.updateOptions({
      readOnly: selection != defaultFacility
    });
    const props = getFacility(selection);
    setRunArgsLeftAddon(props.getRunArgsLeftAddon());
    setRunArgsRightAddon(props.getRunArgsRightAddon());
    setAdditionalRunArgs(props.getDefaultAdditionalRunArgs());
    setInputEditorFileName(props.getInputFileName());
    setOutputEditorFileName(props.getOutputFileName());
    cppEditor.current.setValue(props.getDefaultCodeFile());
    inputEditor.current.setValue(props.getDefaultInputFile());
    outputEditor.current.setValue("");
  }

  const onEditorMounted = (editorRef: React.MutableRefObject<any>) : OnMount => {
    return (editor, _) => {
      editorRef.current = editor;
      if (cppEditor.current && inputEditor.current && outputEditor.current) {
        // All editors mounted.
        setAllEditorsMounted(true);
        setFacilitySelection(defaultFacility);
      }
    }
  }

  const onFacilitySelectionChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setFacilitySelection(event.target.value);
  }

  const onRunButtonClick = () => {
    let input_mlir = inputEditor.current.getValue();
    let printer = (text: string) => {
      setLogValue(currValue => currValue + text + "\n");
    };

    if (currentFacility == defaultFacility) {
      let cpp_source = cppEditor.current.getValue();
      setCompilerState("Compiling...");
      getWasmCompiler()
        .compileAndRun(cpp_source, input_mlir, additionalRunArgs.split(/\s+/), printer)
        .finally(() => { setCompilerState(""); })
        .then((output: string) => { outputEditor.current.setValue(output); }, printer);
    } else {
      let chapterIndex = parseInt(currentFacility.slice(-1));
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
      onMount={onEditorMounted(cppEditor)}
      options={monacoOptions}
    />
  )

  const inputMLIRViewer = (
    <Editor
      key="inputEditor"
      height="100%"
      defaultLanguage="cpp"
      onMount={onEditorMounted(inputEditor)}
      options={monacoOptions}
    />
  )

  const outputMLIRViewer = (
    <Editor
      key="outputEditor"
      height="100%"
      defaultLanguage="cpp"
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
                <Select value={currentFacility} onChange={onFacilitySelectionChange} disabled={!allEditorsMounted}>
                  {
                    getFacilityNames().map((name, i) => {
                      return <option value={name} key={i}>{name}</option>;
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
