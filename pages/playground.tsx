import type { NextPage } from 'next'
import Head from 'next/head'
import Image from 'next/image'
import Editor from '@monaco-editor/react'
import styles from '../styles/Home.module.css'

import {
  Button,
  Grid,
  GridItem,
  Heading,
  VStack,
  Box,
  HStack,
} from '@chakra-ui/react'

import Module from './toyc-ch7.js'

const Playground: NextPage = () => {

  const defaultCode = "Support coming soon."
  const defaultMLIRInput = "Loading..."
  const defaultMLIROutput = defaultMLIRInput

  let wasmInstance = {
    ready: new Promise(resolve => {
      Module({
        onRuntimeInitialized() {
          wasmInstance = Object.assign(this, {
            ready: Promise.resolve(),
            runToy: this.cwrap("main", "number", [])
          });
          resolve(undefined);
        }
      })
    }),
    runToy: undefined, // will be assigned the "main" function of the Toy executable
    inputEditor: undefined, // will be assigned the input monaco editor
    outputEditor: undefined, // will be assigned the output monaco editor
  }

  const onCodeChange = () => {}

  const onInputViewerMount = (editor, _) => {
    wasmInstance.ready.then(() => {
      let default_text = wasmInstance.FS.readFile("input.toy", { encoding: "utf8" });
      editor.setValue(default_text);
      wasmInstance.inputEditor = editor;
    })
  }

  const onOutputViewerMount = (editor, _) => {
    wasmInstance.ready.then(() => {
      editor.setValue("");
      wasmInstance.outputEditor = editor;
    })
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
      key="dslEditor"
      height="100%"
      defaultLanguage="cpp"
      defaultValue={defaultCode}
      onChange={onCodeChange}
      options={Object.assign({}, monacoOptions, {readOnly: true})}
    />
  )

  const inputMLIRViewer = (
    <Editor
      key="dslEditor"
      height="100%"
      defaultLanguage="cpp"
      defaultValue={defaultMLIRInput}
      onMount={onInputViewerMount}
      options={monacoOptions}
    />
  )

  const outputMLIRViewer = (
    <Editor
      key="dslEditor"
      height="100%"
      defaultLanguage="cpp"
      defaultValue={defaultMLIROutput}
      onMount={onOutputViewerMount}
      options={monacoOptions}
    />
  )

  const onRunButtonClick = () => {
    wasmInstance.ready.then(() => {
      if (!wasmInstance.inputEditor || !wasmInstance.outputEditor) {
        return;
      }
      let input_text = wasmInstance.inputEditor.getValue();
      wasmInstance.FS.writeFile("input.toy", input_text, { encoding: "utf8" });
      wasmInstance.runToy();
      let output_text = wasmInstance.FS.readFile("output.mlir", { encoding: "utf8" });
      wasmInstance.outputEditor.setValue(output_text);
    });
  }

  return (
    <div className={styles.container}>
      <Head>
        <title>MLIR Playground</title>
        <meta name="description" content="Playing with MLIR right in the browser." />
        <link rel="icon" href="/mlir.png" />
      </Head>
      <main className={styles.main_playground}>
        <Grid
          templateRows="repeat(2, 1fr)"
          templateColumns="repeat(2, 1fr)"
          columnGap={2}
        >
          <GridItem rowSpan={2} colSpan={1} h="800">
            <VStack spacing={4} align="left" h="100%">
              <HStack>
                <Heading>Editor</Heading>
                <Button
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
          <GridItem rowSpan={1} colSpan={1} h="400" marginTop={1}>
            <Heading>Input</Heading>
            <Box borderWidth="2px" height="100%">
              {inputMLIRViewer}
            </Box>
          </GridItem>
          <GridItem rowSpan={1} colSpan={1} h="400" marginTop={6}>
            <Heading>Output</Heading>
            <Box borderWidth="2px" height="100%">
              {outputMLIRViewer}
            </Box>
          </GridItem>
        </Grid>
      </main>
    </div>
  )
}

export default Playground
