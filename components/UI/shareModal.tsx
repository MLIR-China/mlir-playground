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
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  useToast,
} from "@chakra-ui/react";
import { MdDownload, MdUpload } from "react-icons/md";
import { saveAs } from "file-saver";

import {
  SchemaObjectType,
  validateAgainstSchema,
} from "../Sharing/ImportExport";

export type ShareModalMode = "link" | "file";

export type ShareModalProps = {
  isOpen: boolean;
  mode: ShareModalMode;
  onClose: () => void;
  exportToSchemaObject: () => SchemaObjectType;
  importFromSchemaObject: (source: SchemaObjectType) => void;
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
        const errorMsg = validateAgainstSchema(parsedObject);
        if (!errorMsg) {
          props.importFromSchemaObject(parsedObject as SchemaObjectType);
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
    <Modal isOpen={props.isOpen} onClose={props.onClose} isCentered size="xl">
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
                Share a direct link that others can visit.
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
