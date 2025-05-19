//import React from "react";
import { Box, HStack, Heading, Image, Spacer } from "@chakra-ui/react";

const Header = () => {
  return (
    <Box
      as="header"
      position="fixed"
      top="0"
      left="0"
      right="0"
      zIndex="sticky"
      bg="white"
      boxShadow="sm"
      px={6}
      py={4}
      borderBottom="4px solid"
      borderColor="brand.yellow"
    >
      <HStack spacing={3}>
        <Image src="/vite.svg" alt="EU Flag" boxSize="30px" />
        <Heading size="md" color="brand.blue">
          EU Funding Portal
        </Heading>
        <Spacer />
      </HStack>
    </Box>
  );
};

export default Header
