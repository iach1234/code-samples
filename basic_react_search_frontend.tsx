'use client';

import React, { useState } from 'react';
import styles from './page.module.css';
import Auth from './components/auth';

export default function Home() {
  const [query, setQuery] = useState('');
  const [searchStatus, setSearchStatus] = useState('');
  const [searchResults, setSearchResults] = useState([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSearchStatus('Searching for cases ...');
    setSearchResults([]);

    try {
      const searchHitsResponse = await fetch('/api/search_hits', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const searchHits = await searchHitsResponse.json();

      const topCasesResponse = await fetch('/api/top_cases_progress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reranked_hits_list: searchHits, no_best_hits: 5 }),
      });
      const topCases = await topCasesResponse.json();

      setSearchStatus(topCases);

      const llmOutputResponse = await fetch('/api/llm_output', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reranked_hits_list: searchHits, query, no_best_hits: 5 }),
      });
      const llmOutput = await llmOutputResponse.json();

      const orderedOutput = llmOutput.sort((a, b) => parseInt(b.relevance_score) - parseInt(a.relevance_score));

      setSearchResults(orderedOutput);
      setSearchStatus('');
    } catch (error) {
      console.error('Error:', error);
      setSearchStatus('An error occurred while searching.');
    }
  };

  const renderSearchResult = (hit, index) => {
    const { citation, source, summary, relevance_analysis } = hit;
    const sourceList = source.split(',');

    return (
      <div key={index} className={styles.searchResult}>
        <div className={styles.resultHeader}>
          <span className={styles.resultNumber}>{index + 1}.</span>
          <a href={sourceList[0]} target="_blank" rel="noopener noreferrer" className={styles.resultCitation}>
            {citation}
          </a>
        </div>
        <div className={styles.resultContent}>
          <div className={styles.resultSource}>{source}</div>
          <p className={styles.resultParagraph}>Summary: {summary}</p>
          <p className={styles.resultParagraph}>Relevance: {relevance_analysis}</p>
        </div>
      </div>
    );
  };

  return (
    <Auth>
      <div className={styles.container}>
        <div className={styles.contentWrapper}>
          <div className={styles.headerWrapper}>
            <h1 className={styles.title}><strong>Quick Query</strong></h1>
            <h2 className={styles.subtitle}>Five Hong Kong cases for your review, up to mid-April 2024</h2>
          </div>
          <form onSubmit={handleSubmit} className={styles.form}>
            <textarea
              className={styles.searchBox}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Query (with any additional context)"
              rows={6}
            />
            <button type="submit" className={styles.sendButton}>
              Submit
            </button>
          </form>
          {searchStatus && (
            <div className={styles.searchStatus}>
              {searchStatus.split('\n').map((line, index) => (
                <p key={index}>{line}</p>
              ))}
            </div>
          )}
          <div className={styles.searchResults}>
            {searchResults.map((hit, index) => renderSearchResult(hit, index))}
          </div>
        </div>
      </div>
    </Auth>
  );
}
