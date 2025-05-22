import React, { useEffect, useState } from "react";
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
import type { ProjectExplorerProps, Project, ChatMessage } from "../hooks/types";
import { i } from "framer-motion/client";

interface FilterOptions {
  statuses: string[];
  legalBases: string[];
  organizations: string[];
  countries: string[];
  fundingSchemes: string[];
  ids: string[];
}
const MIN_SEARCH_LEN = 5;

type SortField = keyof Pick<Project, 'title' | 'status' | 'id' | 'startDate' | 'fundingScheme' | 'ecMaxContribution'>;

type SortOrder = 'asc' | 'desc';


const ProjectExplorer: React.FC<ProjectExplorerProps> = ({
  projects,
  search,
  setSearch,
  statusFilter,
  setStatusFilter,
  legalFilter,
  setLegalFilter,
  orgFilter,
  setOrgFilter,
  countryFilter,
  setCountryFilter,
  fundingSchemeFilter,
  setFundingSchemeFilter,
  idFilter,
  setIdFilter,
  setSortField,
  sortField,
  setSortOrder,
  sortOrder,
  page,
  setPage,
  setSelectedProject,
  question,
  setQuestion,
  chatHistory,
  askChatbot,
  messagesEndRef,
}) => {
  const [filterOpts, setFilterOpts] = useState<FilterOptions>({
    statuses: [],
    legalBases: [],
    organizations: [],
    countries: [],
    fundingSchemes: [],
    ids: [],
  });
  const [loadingFilters, setLoadingFilters] = useState(false);

  // Fetch dynamic filter options whenever any filter changes
  useEffect(() => {
    setLoadingFilters(true);
    const params = new URLSearchParams();
    if (statusFilter) params.set("status", statusFilter);
    if (legalFilter)  params.set("legalBasis", legalFilter);
    if (orgFilter)    params.set("organization", orgFilter);
    if (countryFilter) params.set("country", countryFilter);
    if (search)       params.set("search", search);
    if (idFilter.length >= MIN_SEARCH_LEN) params.set("id", idFilter);
    if (fundingSchemeFilter) params.set("fundingScheme", fundingSchemeFilter);
    params.set("sortField", sortField);
    params.set("sortOrder", sortOrder);

    fetch(`/api/filters?${params.toString()}`)
      .then((res) => res.json())
      .then((data: FilterOptions) => setFilterOpts(data))
      .catch(console.error)
      .finally(() => setLoadingFilters(false));
  }, [statusFilter, legalFilter, orgFilter, countryFilter, search, idFilter, fundingSchemeFilter]);
  
  const fmtNum = (num: number | null | undefined): string =>
    num != null
      ? num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
      : '-';
  
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('asc');
    }
    setPage(0);
  };

  const handleMinInput = (value: string, setter: (v: string) => void) => {
    setter(value);
    setPage(0);
  };


  return (
    <Flex direction={{ base: "column", md: "row" }} gap={6}>
      {/* Left Pane: Projects & Filters */}
      <Box w={{ base: "100%", md: "70%" }} p={4}>
        <Flex gap={4} mb={4} flexWrap="wrap">
          <Input
            placeholder="Search by title..."
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(0); }}
            width={{ base: "100%", md: "200px" }}
          />
          <Input
            placeholder={`ID (min ${MIN_SEARCH_LEN})`}
            value={idFilter}
            onChange={(e) => handleMinInput(e.target.value, setIdFilter)}
            w="160px"
            isDisabled={loadingFilters}
          />
          <ChakraSelect
            placeholder={loadingFilters ? "Loading..." : "Status"}
            value={statusFilter}
            onChange={(e) => { setStatusFilter(e.target.value); setPage(0); }}
            isDisabled={loadingFilters}
            width="120px"
          >
            {filterOpts.statuses.map((s) => <option key={s} value={s}>{s}</option>)}
          </ChakraSelect>
          <ChakraSelect
            placeholder={loadingFilters ? "Loading..." : "Legal Basis"}
            value={legalFilter}
            onChange={(e) => { setLegalFilter(e.target.value); setPage(0); }}
            isDisabled={loadingFilters}
            width="150px"
          >
            {filterOpts.legalBases.map((lb) => <option key={lb} value={lb}>{lb}</option>)}
          </ChakraSelect>
          <ChakraSelect
            placeholder={loadingFilters ? "Loading..." : "Organization"}
            value={orgFilter}
            onChange={(e) => { setOrgFilter(e.target.value); setPage(0); }}
            isDisabled={loadingFilters}
            width="150px"
          >
            {filterOpts.organizations.map((o) => <option key={o} value={o}>{o}</option>)}
          </ChakraSelect>
          <ChakraSelect
            placeholder={loadingFilters ? "Loading..." : "Country"}
            value={countryFilter}
            onChange={(e) => { setCountryFilter(e.target.value); setPage(0); }}
            isDisabled={loadingFilters}
            width="120px"
          >
            {filterOpts.countries.map((c) => <option key={c} value={c}>{c}</option>)}
          </ChakraSelect>
          <ChakraSelect
            placeholder={loadingFilters ? "Loading..." : "Funding Scheme"}
            value={fundingSchemeFilter}
            onChange={(e) => { setFundingSchemeFilter(e.target.value); setPage(0); }}
            isDisabled={loadingFilters}
            width="120px"
          >
            {filterOpts.fundingSchemes.map((c) => <option key={c} value={c}>{c}</option>)}
          </ChakraSelect>
        </Flex>

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
                  <Th w="50%" whiteSpace="nowrap" onClick={() => handleSort('title')} cursor="pointer">
                    Title{sortField==='title'? (sortOrder==='asc'?' ↑':' ↓'):''}
                  </Th>
                  <Th w="10%" whiteSpace="nowrap" onClick={() => handleSort('status')} cursor="pointer">
                    Status{sortField==='status'? (sortOrder==='asc'?' ↑':' ↓'):''}
                  </Th>
                  <Th w="10%" whiteSpace="nowrap" onClick={() => handleSort('id')} cursor="pointer">
                    ID{sortField==='id'? (sortOrder==='asc'?' ↑':' ↓'):''}
                  </Th>
                  <Th w="10%" whiteSpace="nowrap" onClick={() => handleSort('startDate')} cursor="pointer">
                    Start Date{sortField==='startDate'? (sortOrder==='asc'?' ↑':' ↓'):''}
                  </Th>
                  <Th w="10%" whiteSpace="nowrap" onClick={() => handleSort('fundingScheme')} cursor="pointer">
                    Funding Scheme{sortField==='fundingScheme'? (sortOrder==='asc'?' ↑':' ↓'):''}
                  </Th>
                  <Th w="10%" whiteSpace="nowrap" onClick={() => handleSort('ecMaxContribution')} cursor="pointer">
                    Funding (€){sortField==='ecMaxContribution'? (sortOrder==='asc'?' ↑':' ↓'):''}
                  </Th>
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
                    <Td w="50%" overflow="hidden" textOverflow="ellipsis">{p.title}</Td>
                    <Td w="10%">{p.status}</Td>
                    <Td w="10%">{p.id}</Td>
                    <Td w="10%" whiteSpace="nowrap">{new Date(p.startDate).toISOString().slice(0,10)}</Td>
                    <Td w="10%">{p.fundingScheme || '-'}</Td>
                    <Td w="10%">€{fmtNum(p.ecMaxContribution)}</Td>

                  </Tr>
                ))}
              </Tbody>
            </Table>
          )}
        </Box>

        <Flex mt={4} gap={2} justify="center">
          <Button onClick={() => setPage(p => Math.max(p - 1, 0))} isDisabled={page === 0}>Previous</Button>
          <Button onClick={() => setPage(p => p + 1)}>Next</Button>
        </Flex>
      </Box>

      {/* Right Pane: Assistant */}
      <Box
        w={{ base: "100%", md: "30%" }}
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
        <Text fontSize="xs" color="gray.500" mb={3}>
          ⚠️ The model may occasionally produce incorrect or misleading answers.
        </Text>
        <Box flex={1} overflowY="auto" mb={4}>
          <VStack spacing={3} align="stretch">
            {chatHistory.map((msg: ChatMessage, i: number) => (
              <HStack
                key={i}
                alignSelf={msg.role === "user" ? "flex-end" : "flex-start"}
                maxW="90%"
              >
                {msg.role === "assistant" && <Avatar size="sm" name="Bot" />}
                <Box>
                  <Text fontSize="sm" bg={msg.role === "user" ? "blue.100" : "gray.200"} px={3} py={2} borderRadius="md">
                    {msg.content}
                  </Text>
                </Box>
                {msg.role === "user" && <Avatar size="sm" name="You" bg="blue.300" />}
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
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); askChatbot(); } }}
          />
          <Button onClick={askChatbot} colorScheme="blue" aria-label="Ask the chatbot">Send</Button>
        </HStack>
      </Box>
    </Flex>
  );
};

export default ProjectExplorer;