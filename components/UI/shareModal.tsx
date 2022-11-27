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
  HStack,
  IconButton,
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
import { MdContentCopy, MdDownload, MdUpload } from "react-icons/md";
import { saveAs } from "file-saver";
import copy from "clipboard-copy";

import { SchemaObjectType } from "../State/ImportExport";

export type ShareModalMode = "link" | "file";

export type ShareModalProps = {
  isOpen: boolean;
  mode: ShareModalMode;
  onClose: () => void;
  exportToSchemaObject: () => SchemaObjectType;
  importFromSchemaObject: (source: any) => string;
};

const createShareLink = (data: string) => {
  return fetch(process.env.shareLinkGenerator!, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: data,
  })
    .then((response) => {
      if (response.ok) {
        return response.json();
      }
      return Promise.reject(
        "Failed to upload state to share link generator. " + response.statusText
      );
    })
    .then((data) => {
      if ("resource" in data) {
        return data["resource"];
      }
      return Promise.reject("Unexpected response from share link generator.");
    });
};

export const ShareModal = (props: ShareModalProps) => {
  const [sharedFileLocation, setSharedFileLocation] =
    React.useState<string>("");
  const onSharedFileLocationChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setSharedFileLocation(event.target.value);
  };

  const [createShareLinkPressed, setCreateShareLinkPressed] =
    React.useState<boolean>(false);
  const closeModal = () => {
    props.onClose();
    // Reset modal state.
    setCreateShareLinkPressed(false);
    setSharedFileLocation("");
  };

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

  const onGetLinkClick = () => {
    setCreateShareLinkPressed(true);
    const schemaObject = props.exportToSchemaObject();
    createShareLink(JSON.stringify(schemaObject)).then(
      (resource) => {
        setSharedFileLocation(resource);
      },
      (error) => {
        toastError("Error creating quick share link.", String(error));
      }
    );
  };

  const onCopyLinkToClipboardClick = (
    event: React.MouseEvent<HTMLButtonElement>
  ) => {
    copy(`${windowLocation}/?import=${sharedFileLocation}`).then(
      () => {
        toast({
          title: "Copied URL to clipboard.",
          status: "success",
          position: "top",
        });
      },
      () => {
        toastError(
          "Error copying URL to clipboard.",
          "Try selecting and copying the URL manually instead."
        );
      }
    );
  };

  return (
    <Modal isOpen={props.isOpen} onClose={closeModal} isCentered size="3xl">
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
                  <Box width="100%">
                    <Text>Share a direct link that others can visit.</Text>

                    <HStack pt={2} width="100%">
                      <IconButton
                        aria-label="Copy to Clipboard"
                        icon={<MdContentCopy />}
                        onClick={onCopyLinkToClipboardClick}
                      ></IconButton>
                      <InputGroup fontFamily="mono" variant="flushed">
                        <InputLeftAddon>{`${windowLocation}/?import=`}</InputLeftAddon>
                        <Input
                          variant="flushed"
                          placeholder="https://example.com/example.json"
                          value={sharedFileLocation}
                          onChange={onSharedFileLocationChange}
                        ></Input>
                      </InputGroup>
                    </HStack>
                  </Box>

                  {process.env.shareLinkGenerator && (
                    <Box width="100%">
                      <Heading as="h4" size="sm">
                        Playground-Hosted Link
                      </Heading>
                      <Text pt={2} pb={2}>
                        Fastest way for sharing temporary work. MLIR Playground
                        will host your code for 48 hours.
                      </Text>
                      <Button
                        disabled={createShareLinkPressed}
                        onClick={onGetLinkClick}
                      >
                        Get Link
                      </Button>
                    </Box>
                  )}

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
                        Host the file somewhere accessible via HTTP GET (such as
                        by{" "}
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
                        Place the resource URL in the text input box above to
                        form the share URL.
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
