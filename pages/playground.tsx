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

const Playground: NextPage = () => {

  const defaultCode = `// type your MLIR lower pass logic here

struct MLIRLoweringPass : public MLIRLoweringBase<MLIRLoweringPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    if (failed(...)) {
      return signalPassFailure();
    }
  }
};
`

  const defaultMLIRInput = `func @dot_general_4x384x32x384() -> tensor<4x384x384xf32> {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<4x384x32xf32>
  %rhs = util.unfoldable_constant dense<1.0> : tensor<4x32x384xf32>
  %0 = "mhlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = {
      lhs_batching_dimensions = dense<0> : tensor<1xi64>,
      lhs_contracting_dimensions = dense<2> : tensor<1xi64>,
      rhs_batching_dimensions = dense<0> : tensor<1xi64>,
      rhs_contracting_dimensions = dense<1> : tensor<1xi64>
      }
  } : (tensor<4x384x32xf32>, tensor<4x32x384xf32>) -> tensor<4x384x384xf32>
  return %0 : tensor<4x384x384xf32>
}`

  const defaultMLIROutput = defaultMLIRInput

  const onCodeChange = () => {}

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
      options={monacoOptions}
    />
  )

  const inputMLIRViewer = (
    <Editor
      key="dslEditor"
      height="100%"
      defaultLanguage="cpp"
      defaultValue={defaultMLIRInput}
      options={monacoOptions}
    />
  )

  const outputMLIRViewer = (
    <Editor
      key="dslEditor"
      height="100%"
      defaultLanguage="cpp"
      defaultValue={defaultMLIROutput}
      options={monacoOptions}
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
