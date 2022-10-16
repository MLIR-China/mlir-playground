import React from "react";
import {
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Box,
  Button,
  ButtonGroup,
  Heading,
  Input,
  InputGroup,
  InputLeftAddon,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  Text,
  useToast,
  VStack,
  Link,
  ListItem,
  OrderedList,
} from "@chakra-ui/react";
import { MdDownload, MdUpload } from "react-icons/md";
import { saveAs } from "file-saver";

import {
  SchemaObjectType,
  validateAgainstSchema,
} from "../Sharing/ImportExport";
import { PlaygroundPreset } from "../Presets/PlaygroundPreset";

export type ShareModalMode = "link" | "file";

export type ShareModalProps = {
  isOpen: boolean;
  mode: ShareModalMode;
  onClose: () => void;
  exportToSchemaObject: () => SchemaObjectType;
  importFromSchemaObject: (source: any) => string;
};

export const ShareModal = (props: ShareModalProps) => {
  const toast = useToast();
  const toastError = (title: string, description: string) => {
    toast({
      title: title,
      description: description,
      status: "error",
      position: "top",
      isClosable: true,
      duration: null,
    });
  };
  const initialExpandedAccordionItemIndex = props.mode == "link" ? 0 : 1;
  const uploadFileInput = React.useRef<HTMLInputElement>(null);

  const [windowLocation, setWindowLocation] = React.useState<string>("");
  React.useEffect(() => {
    setWindowLocation(window.location.origin);
  }, []);

  const onUploadFileInputChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (!event.target.files) {
      toastError("Error uploading from file.", "Failed to find upload file.");
      return;
    }

    const uploadFile = event.target.files[0];
    uploadFile.text().then(
      (rawData) => {
        const parsedObject = JSON.parse(rawData);
        let errorMsg = props.importFromSchemaObject(parsedObject);
        if (!errorMsg) {
          toast({
            title: "Import Success!",
            description: "Successfully imported playground.",
            status: "success",
            position: "top",
          });
          uploadFileInput.current!.value = "";
          props.onClose();
        } else {
          toastError("Error parsing uploaded file.", errorMsg);
        }
      },
      (error) => {
        toastError("Error reading uploaded file.", error);
      }
    );
  };

  const onUploadClick = () => {
    if (uploadFileInput.current) {
      uploadFileInput.current.click();
    }
  };

  const doExport = () => {
    const schemaObject = props.exportToSchemaObject();
    const downloadFile = new Blob([JSON.stringify(schemaObject)], {
      type: "text/plain;charset=utf-8",
    });
    saveAs(downloadFile, "playground.json");
  };

  return (
    <Modal isOpen={props.isOpen} onClose={props.onClose} isCentered size="3xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Share your playground with others</ModalHeader>
        <ModalCloseButton />

        <ModalBody>
          <Accordion defaultIndex={initialExpandedAccordionItemIndex}>
            <AccordionItem>
              <h2>
                <AccordionButton>
                  <Box flex="1" textAlign="left">
                    Share Link
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
              </h2>
              <AccordionPanel pb={4}>
                <VStack spacing={4} align="start">
                  <Box>
                    <Text>Share a direct link that others can visit.</Text>
                  </Box>

                  <Box>
                    <Heading as="h4" size="sm">
                      Playground-Hosted Link
                    </Heading>
                    <Text pt={2} pb={2}>
                      Fastest way for sharing temporary work. MLIR Playground
                      will host your code for 48 hours.
                    </Text>
                    <Button>Get Link</Button>
                  </Box>

                  <Box width="100%">
                    <Heading as="h4" size="sm">
                      Self-Hosted Link
                    </Heading>
                    <Text pt={2} pb={2}>
                      Self-host your code without size or time limits. Best for
                      demos or tutorials.
                    </Text>
                    <Text>Steps:</Text>
                    <OrderedList>
                      <ListItem>
                        Use the &quot;Export&quot; feature to save this
                        playground as a JSON file.
                      </ListItem>
                      <ListItem>
                        Host the file somewhere accessible via HTTP (such as by{" "}
                        <Link
                          href="https://gist.github.com/"
                          isExternal
                          color="blue.400"
                        >
                          creating a GitHub Gist
                        </Link>
                        ).
                      </ListItem>
                      <ListItem>
                        Share the combined link:
                        <br />
                        <InputGroup fontFamily="mono" variant="flushed">
                          <InputLeftAddon>{`${windowLocation}?import=`}</InputLeftAddon>
                          <Input variant="flushed"></Input>
                        </InputGroup>
                      </ListItem>
                    </OrderedList>
                  </Box>
                </VStack>
              </AccordionPanel>
            </AccordionItem>

            <AccordionItem>
              <h2>
                <AccordionButton>
                  <Box flex="1" textAlign="left">
                    Export/Import File
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
              </h2>
              <AccordionPanel pb={4}>
                <Box pb={4}>
                  <p>Export a sharable file, or import an exported file.</p>
                  <p>(Note: Importing will overwrite your current work.)</p>
                </Box>

                <ButtonGroup colorScheme="blue">
                  <Button leftIcon={<MdDownload />} onClick={doExport}>
                    Export
                  </Button>
                  <Button leftIcon={<MdUpload />} onClick={onUploadClick}>
                    Import
                  </Button>
                </ButtonGroup>

                <input
                  ref={uploadFileInput}
                  type="file"
                  style={{ display: "none" }}
                  onChange={onUploadFileInputChange}
                />
              </AccordionPanel>
            </AccordionItem>
          </Accordion>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
};
