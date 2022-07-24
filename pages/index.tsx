import React, { useRef, useState } from "react";
import type { NextPage } from "next";
import Head from "next/head";

import {
  Box,
  Button,
  Flex,
  Grid,
  GridItem,
  Heading,
  HStack,
  Image,
  Input,
  InputGroup,
  InputLeftAddon,
  InputRightAddon,
  Select,
  Spacer,
  Text,
  Textarea,
  VStack,
} from "@chakra-ui/react";
import Editor, { OnMount } from "@monaco-editor/react";
import styles from "../styles/Home.module.css";

import {
  defaultFacility,
  getFacilityNames,
  getFacility,
} from "../components/Facilities/FacilitySelector";

const Home: NextPage = () => {
  const monospaceFontFamily = "Consolas, 'Courier New', monospace";

  // state
  const [allEditorsMounted, setAllEditorsMounted] = useState(false);
  const cppEditor: React.MutableRefObject<any> = useRef(null);
  const inputEditor: React.MutableRefObject<any> = useRef(null);
  const outputEditor: React.MutableRefObject<any> = useRef(null);
  const [logValue, setLogValue] = useState("");

  const [currentFacility, setCurrentFacility] = React.useState(defaultFacility);

  const [runArgsLeftAddon, setRunArgsLeftAddon] = useState("");
  const [runArgsRightAddon, setRunArgsRightAddon] = useState("");
  const [additionalRunArgs, setAdditionalRunArgs] = useState("");
  const [inputEditorFileName, setInputEditorFileName] = useState("");
  const [outputEditorFileName, setOutputEditorFileName] = useState("");

  const [runStatus, setRunStatus] = useState("");

  function setFacilitySelection(selection: string) {
    setCurrentFacility(selection);
    const props = getFacility(selection);
    cppEditor.current.updateOptions({
      readOnly: !props.isCodeEditorEnabled(),
    });
    setRunArgsLeftAddon(props.getRunArgsLeftAddon());
    setRunArgsRightAddon(props.getRunArgsRightAddon());
    setAdditionalRunArgs(props.getDefaultAdditionalRunArgs());
    setInputEditorFileName(props.getInputFileName());
    setOutputEditorFileName(props.getOutputFileName());
    cppEditor.current.setValue(props.getDefaultCodeFile());
    inputEditor.current.setValue(props.getDefaultInputFile());
    outputEditor.current.setValue("");
  }

  const onEditorMounted = (editorRef: React.MutableRefObject<any>): OnMount => {
    return (editor, _) => {
      editorRef.current = editor;
      if (cppEditor.current && inputEditor.current && outputEditor.current) {
        // All editors mounted.
        setAllEditorsMounted(true);
        setFacilitySelection(defaultFacility);
      }
    };
  };

  const onFacilitySelectionChange = (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    setFacilitySelection(event.target.value);
  };

  const onRunButtonClick = () => {
    const input_mlir = inputEditor.current.getValue();
    const printer = (text: string) => {
      setLogValue((currValue) => currValue + text + "\n");
    };

    const facility = getFacility(currentFacility);
    let cpp_source = "";
    if (facility.isCodeEditorEnabled()) {
      cpp_source = cppEditor.current.getValue();
      setRunStatus("Compiling...");
    } else {
      setRunStatus("Running...");
    }
    facility
      .run(cpp_source, input_mlir, additionalRunArgs, printer)
      .finally(() => {
        setRunStatus("");
      })
      .then((output: string) => {
        outputEditor.current.setValue(output);
      }, printer);
  };

  const monacoOptions = {
    selectOnLineNumbers: true,
    quickSuggestions: true,
    minimap: {
      enabled: false,
    },
    scrollBeyondLastLine: false,
    automaticLayout: true,
  };

  const codeEditor = (
    <Editor
      key="cppEditor"
      height="100%"
      defaultLanguage="cpp"
      onMount={onEditorMounted(cppEditor)}
      options={monacoOptions}
    />
  );

  const inputMLIRViewer = (
    <Editor
      key="inputEditor"
      height="100%"
      defaultLanguage="cpp"
      onMount={onEditorMounted(inputEditor)}
      options={monacoOptions}
    />
  );

  const outputMLIRViewer = (
    <Editor
      key="outputEditor"
      height="100%"
      defaultLanguage="cpp"
      onMount={onEditorMounted(outputEditor)}
      options={{ ...monacoOptions, readOnly: true }}
    />
  );

  return (
    <div className={styles.container}>
      <Head>
        <title>MLIR Playground</title>
        <meta
          name="description"
          content="Play with MLIR right in the browser."
        />
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon/favicon-32x32.png" />
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon/favicon-16x16.png" />
        <link rel="apple-touch-icon" sizes="180x180" href="/favicon/apple-touch-icon.png" />
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
                <Image src="/mlir-playground.png" alt="MLIR Playground" boxSize="2em" />
                <Heading>MLIR Playground</Heading>
                <Button
                  isLoading={!allEditorsMounted || runStatus !== ""}
                  mt="8"
                  as="a"
                  size="lg"
                  colorScheme="blue"
                  fontWeight="bold"
                  onClick={onRunButtonClick}
                >
                  Run
                </Button>
                <Text>{runStatus}</Text>
              </HStack>
              <Box>
                <Select
                  value={currentFacility}
                  onChange={onFacilitySelectionChange}
                  disabled={!allEditorsMounted}
                >
                  {getFacilityNames().map((name, i) => {
                    return (
                      <option value={name} key={i}>
                        {name}
                      </option>
                    );
                  })}
                </Select>
              </Box>
              <HStack>
                <Text>Arguments</Text>
                <InputGroup fontFamily={monospaceFontFamily}>
                  <InputLeftAddon>{runArgsLeftAddon}</InputLeftAddon>
                  <Input
                    value={additionalRunArgs}
                    onChange={(event) =>
                      setAdditionalRunArgs(event.target.value)
                    }
                  ></Input>
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
              <Text fontFamily={monospaceFontFamily}>
                {inputEditorFileName}
              </Text>
            </Flex>
            <Box borderWidth="2px" h="200">
              {inputMLIRViewer}
            </Box>
          </GridItem>
          <GridItem rowSpan={1} colSpan={1}>
            <Flex align="end">
              <Heading>Output</Heading>
              <Spacer />
              <Text fontFamily={monospaceFontFamily}>
                {outputEditorFileName}
              </Text>
            </Flex>
            <Box borderWidth="2px" h="200">
              {outputMLIRViewer}
            </Box>
          </GridItem>
          <GridItem rowSpan={2} colSpan={1}>
            <Heading>Logs</Heading>
            <Textarea
              borderWidth="2px"
              height="100%"
              bg="gray.800"
              value={logValue}
              readOnly
              color="white"
              fontFamily={monospaceFontFamily}
            ></Textarea>
          </GridItem>
        </Grid>
      </main>
    </div>
  );
};

export default Home;
