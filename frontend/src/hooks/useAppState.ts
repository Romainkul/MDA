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
  const [loading, setLoading] = useState<boolean>(false);

  const [availableFilters, setAvailableFilters] = useState<AvailableFilters>({
    statuses: ["SIGNED", "CLOSED", "TERMINATED","UNKNOWN"],
    organizations: [],
    countries: [],
    legalBases: []
  });

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const fetchProjects = () => {
    fetch(`/api/projects?page=${page}&search=${encodeURIComponent(search)}&status=${statusFilter}&legalBasis=${legalFilter}&organization=${orgFilter}&country=${countryFilter}&fundingScheme=${fundingSchemeFilter}&proj_id=${idFilter}&sortField=${sortField}&sortOrder=${sortOrder}`)
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
          //ids: data.ids
        });
      });
  };
  interface RagResponse {
    answer: string;
    source_ids: string[];
  }

  const askChatbot = async () => {
    if (!question.trim() || loading) return;          
    const newChat: ChatMessage[] = [
      ...chatHistory,
      { role: "user", content: question },
    ];
    setChatHistory(newChat);
    setQuestion("");
    setLoading(true);                                      

    // 1) placeholder
    setChatHistory((h) => [
      ...h,
      { role: "assistant", content: "Generating answer..." },
    ]);

    try {
      const res = await fetch("/api/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question }),
      });

      const text = await res.text();
      if (!res.ok) {
        let errDetail = text;
        try {
          errDetail = JSON.parse(text).detail;
        } catch {}
        throw new Error(errDetail);
      }

      const data: RagResponse = JSON.parse(text);
      const idList = data.source_ids.join(", ") || "none";
      const assistantContent = `${data.answer}

The output was based on the following Project IDs: ${idList}`;

      // 2) replace placeholder with real answer
      setChatHistory((h) => [
        ...h.slice(0, -1),
        { role: "assistant", content: assistantContent },
      ]);
    } catch (err: any) {
      // replace placeholder with error message
      setChatHistory((h) => [
        ...h.slice(0, -1),
        {
          role: "assistant",
          content: `Something went wrong: ${err.message}`,
        },
      ]);
    } finally {
      setLoading(false);
      // scroll to bottom
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  };

  useEffect(() => {
    // If the user has typed something but it's too short, don't refetch
    if (search.length > 0 && search.length < 3) {
      return;
    }
    fetchProjects();
  }, [
    page,
    search,
    statusFilter,
    legalFilter,
    orgFilter,
    countryFilter,
    fundingSchemeFilter,
    idFilter,
    sortField,
    sortOrder,
  ]);
  
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
      loading,
      messagesEndRef
    },
    detailsProps: {
      project: selectedProject!,
      question,
      setQuestion,
      chatHistory,
      askChatbot,
      loading,
      messagesEndRef
    }
  };
};
