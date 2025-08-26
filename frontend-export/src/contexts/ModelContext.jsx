
import React, { createContext, useState, useContext, useMemo, useCallback } from 'react';
import { supabase } from '@/lib/customSupabaseClient';
import { useToast } from '@/components/ui/use-toast';

const ModelContext = createContext();

const modelsData = {
  'claude-3-haiku-20240307': {
    name: 'Sonnet (Haiku)',
    description: 'A balanced model for lyrical creativity and structure.',
  },
  'claude-3-opus-20240229': {
    name: 'Opus',
    description: 'The most powerful model, for complex and nuanced compositions.',
  },
};

export function ModelProvider({ children }) {
  const { toast } = useToast();
  const [selectedModel, setSelectedModel] = useState('claude-3-haiku-20240307');
  const [isGenerating, setIsGenerating] = useState(false);

  const generateSong = useCallback(async (promptDetails, accessToken) => {
    setIsGenerating(true);

    try {
      const { data, error } = await supabase.functions.invoke('generate-song', {
        body: JSON.stringify({
          prompt: promptDetails.genrePrompt,
          lyrics: promptDetails.lyricsValue,
          model: selectedModel,
          user_provided_title: promptDetails.userProvidedTitle,
          use_ai_lyrics: promptDetails.useAILyrics,
          is_instrumental: promptDetails.lyricsValue === '[Instrumental]'
        }),
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });

      if (error) {
        throw new Error(error.message);
      }
      
      if (data.error) {
        const errorMessage = typeof data.error === 'object' ? JSON.stringify(data.error) : data.error;
        throw new Error(errorMessage);
      }
      
      return {
        title: data.title,
        lyrics: data.lyrics,
        model: data.model,
        audio_url: data.audio_url
      };

    } catch (error) {
      console.error('Error generating song:', error);
      let description = 'Something went wrong while creating your song.';
      try {
        const parsedError = JSON.parse(error.message);
        description = parsedError.error || parsedError.detail || description;
      } catch (e) {
        description = error.message || description;
      }
      toast({
        title: 'Generation Failed',
        description: description,
        variant: 'destructive',
      });
      return null;
    } finally {
      setIsGenerating(false);
    }
  }, [selectedModel, toast]);

  const value = useMemo(() => ({
    models: modelsData,
    selectedModel,
    setSelectedModel,
    isGenerating,
    generateSong,
  }), [selectedModel, isGenerating, generateSong]);

  return (
    <ModelContext.Provider value={value}>
      {children}
    </ModelContext.Provider>
  );
}

export const useModel = () => {
  const context = useContext(ModelContext);
  if (context === undefined) {
    throw new Error('useModel must be used within a ModelProvider');
  }
  return context;
};
