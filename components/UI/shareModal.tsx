import React, { useState } from "react";
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
} from "@chakra-ui/react";
import { MdDownload, MdUpload } from "react-icons/md";

export type ShareModalMode = "link" | "file";

export type ShareModalProps = {
  isOpen: boolean;
  mode: ShareModalMode;
  onClose: () => void;
};

export const ShareModal = (props: ShareModalProps) => {
  const initialExpandedAccordionItemIndex = props.mode == "link" ? 0 : 1;

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
                  <Button leftIcon={<MdDownload />}>Export</Button>
                  <Button leftIcon={<MdUpload />}>Import</Button>
                </ButtonGroup>
              </AccordionPanel>
            </AccordionItem>
          </Accordion>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
};
