import React from "react";
import {
  Box,
  Button,
  Flex,
  Heading,
  Input,
  Select as ChakraSelect,
  Spinner,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
  VStack,
  HStack,
  Text,
  Avatar,
} from "@chakra-ui/react";
import type {
  ProjectExplorerProps,
  Project,
  ChatMessage,
} from "../hooks/types";

const ProjectExplorer: React.FC<ProjectExplorerProps> = ({
  projects,
  search,
  setSearch,
  statusFilter,
  setStatusFilter,
  page,
  setPage,
  setSelectedProject,
  question,
  setQuestion,
  chatHistory,
  askChatbot,
  messagesEndRef,
}) => (
  <Flex direction={{ base: "column", md: "row" }} gap={6}>
    {/* Left Pane: Projects */}
    <Box flex={1}>
      <Heading size="sm" mb={2}>
        Projects
      </Heading>
      <Flex gap={4} mb={4}>
        <Input
          placeholder="Search by title..."
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(0);
          }}
        />
        <ChakraSelect
          placeholder="Filter by status"
          value={statusFilter}
          onChange={(e) => {
            setStatusFilter(e.target.value);
            setPage(0);
          }}
        >
          <option value="Signed">Signed</option>
          <option value="Closed">Closed</option>
          <option value="Terminated">Terminated</option>
        </ChakraSelect>
      </Flex>

      {/* Table Container with grey background */}
        <Box
          bg="gray.50"
          p={4}
          borderRadius="md"
          height="500px"
          overflowY="auto"
        >
        {!projects.length ? (
          <Flex justify="center" py={10}>
            <Spinner />
          </Flex>
        ) : (
          <Table
                variant="simple"
                size="sm"   
                width="100%"             
            >
            <Thead>
              <Tr>
                <Th width="60%" whiteSpace="nowrap">Title</Th>
                <Th width="10%" whiteSpace="nowrap">Status</Th>
                <Th width="10%" whiteSpace="nowrap">ID</Th>
                <Th width="10%" whiteSpace="nowrap">Start Date</Th>
                <Th width="10%" whiteSpace="nowrap">Funding €</Th>
              </Tr>
            </Thead>
            <Tbody>
              {projects.map((p: Project) => (
                <Tr
                  key={p.id}
                  onClick={() => setSelectedProject(p)}
                  cursor="pointer"
                  _hover={{ bg: "gray.100" }}
                >
                  <Td overflow="hidden" textOverflow="ellipsis">{p.title}</Td>
                  <Td>{p.status}</Td>
                  <Td>{p.id}</Td>
                  <Td whiteSpace="nowrap">{p.startDate}</Td>
                  <Td>{p.ecMaxContribution?.toLocaleString()}</Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        )}
      </Box>

      <Flex mt={4} gap={2} justify="center">
        <Button
          onClick={() => setPage((p) => Math.max(p - 1, 0))}
          isDisabled={page === 0}
          aria-label="Previous page"
        >
          Previous
        </Button>
        <Button onClick={() => setPage((p) => p + 1)} aria-label="Next page">
          Next
        </Button>
      </Flex>
    </Box>

    {/* Right Pane: Assistant */}
    <Box
      flex={{ base: 1, md: 0.4 }}
      bg="gray.50"
      p={4}
      borderRadius="md"
      height="500px"
      display="flex"
      flexDirection="column"
    >
      <Heading size="sm" mb={2}>
        Assistant
      </Heading>
      <Box flex={1} overflowY="auto" mb={4}>
        <VStack spacing={3} align="stretch">
          {chatHistory.map((msg: ChatMessage, i: number) => (
            <HStack
              key={i}
              alignSelf={msg.role === "user" ? "flex-end" : "flex-start"}
              maxW="90%"
            >
              {msg.role === "assistant" && (
                <Avatar size="sm" name="Bot" />
              )}
              <Box>
                <Text
                  fontSize="sm"
                  bg={msg.role === "user" ? "blue.100" : "gray.200"}
                  px={3}
                  py={2}
                  borderRadius="md"
                >
                  {msg.content}
                </Text>
              </Box>
              {msg.role === "user" && (
                <Avatar size="sm" name="You" bg="blue.300" />
              )}
            </HStack>
          ))}
          <div ref={messagesEndRef} />
        </VStack>
      </Box>
      <HStack>
        <Input
          placeholder="Ask something..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              askChatbot();
            }
          }}
        />
        <Button
          onClick={askChatbot}
          colorScheme="blue"
          aria-label="Ask the chatbot"
        >
          Send
        </Button>
      </HStack>
    </Box>
  </Flex>
);

export default ProjectExplorer;
