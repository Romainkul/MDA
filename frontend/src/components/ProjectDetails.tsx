import { useEffect, useState } from "react";//React,
import {
  Box,
  Flex,
  Heading,
  Text,
  Spinner,
  SimpleGrid,
  Badge,
  Wrap,
  Tag,
  Divider,
  VStack,
  HStack,
  Input,
  Button,
  Avatar,
} from "@chakra-ui/react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { ProjectDetailsProps, OrganizationLocation } from "../hooks/types";

import markerIconPng from "leaflet/dist/images/marker-icon.png";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

const customIcon = new L.Icon({
  iconUrl: markerIconPng,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});

function ResizeMap({ count }: { count: number }) {
  const map = useMap();
  useEffect(() => {
    map?.invalidateSize();
  }, [count, map]);
  return null;
}

export default function ProjectDetails({
  project,
  question,
  setQuestion,
  askChatbot,
  chatHistory = [],
  messagesEndRef,
}: ProjectDetailsProps) {
  // fetch organization locations
  const [orgLocations, setOrgLocations] = useState<OrganizationLocation[]>([]);
  const [loadingOrgs, setLoadingOrgs] = useState(true);

  useEffect(() => {
    if (!project) return;
    setLoadingOrgs(true);
    fetch(`/api/project/${project.id}/organizations`)
      .then((r) => r.json())
      .then((data) => Array.isArray(data) ? setOrgLocations(data) : console.error(data))
      .catch(console.error)
      .finally(() => setLoadingOrgs(false));
  }, [project]);

  if (!project) {
    return (
      <Box p={6} textAlign="center">
        <Text color="gray.500">No project selected.</Text>
      </Box>
    );
  }

  // Map center fallback
  const center: [number, number] = orgLocations.length
    ? [orgLocations[0].latitude, orgLocations[0].longitude]
    : [51.505, -0.09];

  return (
    <Flex direction={{ base: "column", md: "row" }} gap={8}>
      {/* Left: Details */}
      <Box flex={1} p={6} bg="white" borderRadius="md" boxShadow="sm">
        <HStack mb={4} align="baseline">
          <Heading size="l">{project.title}</Heading>
          <Badge colorScheme={project.status === "CLOSED" ? "green" : "red"}>
            {project.status}
          </Badge>
        </HStack>

        <SimpleGrid columns={[1, 2]} spacing={4} mb={6}>
          <Box>
            <Text fontWeight="bold">ID</Text>
            <Text>{project.id}</Text>
          </Box>
          <Box>
            <Text fontWeight="bold">Acronym</Text>
            <Text>{project.acronym}</Text>
          </Box>
          <Box>
            <Text fontWeight="bold">Start Date</Text>
            <Text>{project.startDate}</Text>
          </Box>
          <Box>
            <Text fontWeight="bold">End Date</Text>
            <Text>{project.endDate}</Text>
          </Box>
          <Box>
            <Text fontWeight="bold">Funding (EC max)</Text>
            <Text>€{project.ecMaxContribution.toLocaleString()}</Text>
          </Box>
          <Box>
            <Text fontWeight="bold">Legal Basis</Text>
            <Text>{project.legalBasis}</Text>
          </Box>
          <Box gridColumn={["auto", "span 2"]}>
            <Text fontWeight="bold">Framework Programme</Text>
            <Text>{project.frameworkProgramme}</Text>
          </Box>
        </SimpleGrid>

        <Box mb={6}>
          <Heading size="md" mb={2}>Objective</Heading>
          <Text whiteSpace="pre-wrap">{project.objective}</Text>
        </Box>

        {(project.list_euroSciVocTitle ?? []).length > 0 && (
          <Box mb={4}>
            <Heading size="md" mb={2}>EuroSciVoc Titles</Heading>
            <Wrap>
              {(project.list_euroSciVocTitle ?? []).map((t) => (
                <Tag key={t} mb={2}>{t}</Tag>
              ))}
            </Wrap>
          </Box>
        )}

        {(project.list_euroSciVocPath ?? []).length > 0 && (
          <Box mb={4}>
            <Heading size="md" mb={2}>EuroSciVoc Paths</Heading>
            <Wrap>
              {(project.list_euroSciVocPath ?? []).map((p) => (
                <Tag key={p} mb={2}>{p}</Tag>
              ))}
            </Wrap>
          </Box>
        )}

        <Divider my={6} />

        <Heading size="md" mb={3}>Participating Organizations</Heading>
        {loadingOrgs ? (
          <Spinner />
        ) : (
          <Box w="100%" h="300px" borderRadius="md" overflow="hidden">
            <MapContainer center={center} zoom={4} style={{ height: '100%', width: '100%' }}>
              <ResizeMap count={orgLocations.length} />
              <TileLayer
                attribution="&copy; OpenStreetMap contributors"
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              {orgLocations.map((org, i) => (
                <Marker key={i} position={[org.latitude, org.longitude]} icon={customIcon}>
                  <Popup>
                    <Text fontWeight="bold">{org.name}</Text>
                    <Text>{org.country}</Text>
                  </Popup>
                </Marker>
              ))}
            </MapContainer>
          </Box>
        )}
      </Box>

      {/* Right: Chatbot */}
      <Box
        flex={{ base: '1', md: '0.6' }}
        bg="gray.50"
        p={4}
        borderRadius="md"
        display="flex"
        flexDirection="column"
        maxH="600px"
      >
        <Heading size="sm" mb={2}>Ask about this project</Heading>

        <Box flex={1} overflowY="auto" mb={4}>
          <VStack spacing={3} align="stretch">
            {(chatHistory ?? []).map((msg, i) => (
              <HStack
                key={i}
                alignSelf={msg.role === "user" ? "flex-end" : "flex-start"}
                maxW="90%"
              >
                {msg.role === "assistant" && <Avatar size="sm" name="Bot" />}
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
                {msg.role === "user" && <Avatar size="sm" name="You" bg="blue.300" />}
              </HStack>
            ))}
            <div ref={messagesEndRef} />
          </VStack>
        </Box>

        <HStack>
          <Input
            placeholder="Type your question..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                askChatbot();
              }
            }}
          />
          <Button onClick={askChatbot} aria-label="Send question">
            Send
          </Button>
        </HStack>
      </Box>
    </Flex>
  );
}
