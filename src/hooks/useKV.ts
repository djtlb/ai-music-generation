/**
 * React hook for accessing the auto-mixing system functionality.
 */

import { useState, useEffect } from 'react';

export function useKV<T>(key: string, defaultValue: T): [T, (value: T | ((prev: T) => T)) => void, () => void] {
  const [value, setValue] = useState<T>(defaultValue);

  // Load value from KV storage on mount
  useEffect(() => {
    const loadValue = async () => {
      try {
        const stored = await spark.kv.get<T>(key);
        if (stored !== undefined) {
          setValue(stored);
        }
      } catch (error) {
        console.warn(`Failed to load KV value for key "${key}":`, error);
      }
    };
    loadValue();
  }, [key]);

  // Update function that also persists to KV
  const updateValue = async (newValue: T | ((prev: T) => T)) => {
    const nextValue = typeof newValue === 'function' ? (newValue as (prev: T) => T)(value) : newValue;
    setValue(nextValue);
    try {
      await spark.kv.set(key, nextValue);
    } catch (error) {
      console.warn(`Failed to save KV value for key "${key}":`, error);
    }
  };

  // Delete function
  const deleteValue = async () => {
    setValue(defaultValue);
    try {
      await spark.kv.delete(key);
    } catch (error) {
      console.warn(`Failed to delete KV value for key "${key}":`, error);
    }
  };

  return [value, updateValue, deleteValue];
}