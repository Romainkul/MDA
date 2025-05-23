import { useState, useEffect, useRef } from "react";
import { debounce } from "lodash";
import type { Project, OrganizationLocation, FilterState, ChatMessage,AvailableFilters } from "./types";

interface Stats {
  [key: string]: {
    labels: string[];
    values: number[];
  };
}

type SortOrder = "asc" | "desc";

export const useAppState = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [search, setSearch] = useState<string>("");
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [page, setPage] = useState<number>(0);
  const [question, setQuestion] = useState<string>("");
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [stats, setStats] = useState<Stats>({});
  const [legalFilter, setLegalFilter] = useState('');
  const [orgFilter, setOrgFilter] = useState('');
  const [countryFilter, setCountryFilter] = useState('');
  const [fundingSchemeFilter, setFundingSchemeFilter ] = useState('');
  const [idFilter, setIdFilter] = useState('');
  const [sortField, setSortField] = useState('');
  const [sortOrder, setSortOrder] = useState<SortOrder>("asc");
  const [filters, setFilters] = useState<FilterState>({
    status: "",
    organization: "",
    country: "",
    legalBasis: "",
    minYear: "2000",
    maxYear: "2025",
    minFunding: "0",
    maxFunding: "10000000",
    });

  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);

  const [availableFilters, setAvailableFilters] = useState<AvailableFilters>({
    statuses: ["SIGNED", "CLOSED", "TERMINATED","UNKNOWN"],
    organizations: [],
    countries: [],
    legalBases: []
  });

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const fetchProjects = () => {
    fetch(`/api/projects?page=${page}&search=${encodeURIComponent(search)}&status=${statusFilter}&legalBasis=${legalFilter}&organization=${orgFilter}&country=${countryFilter}&fundingScheme=${fundingSchemeFilter}&id=${idFilter}&sortField=${sortField}&sortOrder=${sortOrder}`)
      .then(res => res.json())
      .then((data: Project[]) => setProjects(data))
      .catch(console.error);
  };

  const fetchStats = debounce((filters: FilterState) => {
    const params = new URLSearchParams(filters);
    fetch(`/api/stats?${params.toString()}`)
      .then(res => res.json())
      .then((data: Stats) => setStats(data))
      .catch(console.error);
  }, 500);

  const fetchAvailableFilters = (filters: FilterState) => {
    const params = new URLSearchParams(filters);
    fetch(`/api/filters?${params.toString()}`)
      .then(res => res.json())
      .then((data: Omit<AvailableFilters, 'statuses'>) => {
        setAvailableFilters({
          statuses: ["SIGNED", "CLOSED", "TERMINATED", "UNKNOWN"],
          organizations: data.organizations,
          countries: data.countries,
          legalBases: data.legalBases,
          fundingSchemes: data.fundingSchemes,
          ids: data.ids
        });
      });
  };
  interface RagResponse {
    answer: string;
    source_ids: string[];
  }

  const askChatbot = async () => {
    if (!question.trim()) return;
    const newChat: ChatMessage[] = [
      ...chatHistory,
      { role: "user", content: question },
    ];
    setChatHistory(newChat);
    setQuestion("");

    try {
      const res = await fetch("/api/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question }),
      });

      // Log the raw response for debugging
      console.log("RAG API status:", res.status, res.statusText);
      const text = await res.text();
      console.log("RAG API raw body:", text);

      if (!res.ok) {
        // Try to parse JSON error, or fall back to raw text
        let errDetail: string;
        try {
          const errJson = JSON.parse(text);
          errDetail = errJson.detail || JSON.stringify(errJson);
        } catch {
          errDetail = text;
        }
        throw new Error(`API error ${res.status}: ${errDetail}`);
      }

      // Now parse the successful JSON
      const data: RagResponse = JSON.parse(text);
      console.log("RAG API parsed:", data);

      const idList = data.source_ids.join(", ") || "none";
      const assistantContent = `${data.answer}

  The output was based on the following Project IDs: ${idList}`;

      setChatHistory([
        ...newChat,
        { role: "assistant", content: assistantContent },
      ]);
    } catch (err: any) {
      console.error("askChatbot error:", err);
      setChatHistory([
        ...newChat,
        {
          role: "assistant",
          content: `Something went wrong: ${err.message}`,
        },
      ]);
    }
  };


  useEffect(fetchProjects, [page, search, statusFilter,legalFilter, orgFilter, countryFilter, fundingSchemeFilter, idFilter, sortField, sortOrder]);
  useEffect(() => {
    console.log("Updated filters:", filters);
    fetchStats(filters);
    }, [filters]);
  useEffect(() => fetchAvailableFilters(filters), [filters]);

  return {
    selectedProject,
    dashboardProps: {
      stats,
      filters,
      setFilters,
      availableFilters
    },
    explorerProps: {
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
      setChatHistory,
      askChatbot,
      messagesEndRef
    },
    detailsProps: {
      project: selectedProject!,
      question,
      setQuestion,
      chatHistory,
      askChatbot,
      messagesEndRef
    }
  };
};
