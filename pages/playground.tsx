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

  const defaultCode = "fjsdlak;\nfjasd;\nlfals;\njfiopajfpoasdjfopiasd"

  const defaultMLIRInput = ""
  const defaultMLIROutput = ""

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
      defaultLanguage="python"
      defaultValue={defaultCode}
      onChange={onCodeChange}
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
              <Box borderWidth="1px" height="100%">
                {codeEditor}
              </Box>
            </VStack>
          </GridItem>
        </Grid>
      </main>
    </div>
  )
}

export default Playground
