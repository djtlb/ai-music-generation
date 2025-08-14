import { useState, useEffect, useCallback } from 'react';

/**
 * React hook for persistent key-value storage using the Spark runtime
 * @param key - Unique key for the stored value
 * @param defaultValue - Default value to use if key doesn't exist
 * @returns [value, setValue, deleteValue] tuple
 */
export function useKV<T>(
  key: string,
  defaultValue: T
): [T, (value: T | ((prev: T) => T)) => void, () => void] {
  const [value, setValue] = useState<T>(defaultValue);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load initial value from KV store
  useEffect(() => {
    const loadValue = async () => {
      try {
        const stored = await spark.kv.get<T>(key);
        if (stored !== undefined) {
          setValue(stored);
        }
        setIsLoaded(true);
      } catch (error) {
        console.error(`Failed to load KV value for key "${key}":`, error);
        setIsLoaded(true);
      }
    };

    loadValue();
  }, [key]);

  // Update function that also persists to KV store
  const updateValue = useCallback(
    async (newValue: T | ((prev: T) => T)) => {
      try {
        const valueToSet = typeof newValue === 'function' ? (newValue as (prev: T) => T)(value) : newValue;
        setValue(valueToSet);
        await spark.kv.set(key, valueToSet);
      } catch (error) {
        console.error(`Failed to save KV value for key "${key}":`, error);
      }
    },
    [key, value]
  );

  // Delete function that removes from KV store
  const deleteValue = useCallback(async () => {
    try {
      setValue(defaultValue);
      await spark.kv.delete(key);
    } catch (error) {
      console.error(`Failed to delete KV value for key "${key}":`, error);
    }
  }, [key, defaultValue]);

  // Return default value until loaded to prevent hydration issues
  return [isLoaded ? value : defaultValue, updateValue, deleteValue];
}