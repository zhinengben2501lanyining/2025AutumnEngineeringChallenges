"use client"
import { useState } from 'react';
import { GlobeAsiaAustraliaIcon as Globe, ArrowRightIcon as ArrowRight, ClipboardDocumentListIcon as Copy, CheckIcon as Check, ArrowPathRoundedSquareIcon as RefreshCw, LanguageIcon as Language } from '@heroicons/react/24/outline';  
import {Spinner} from "@heroui/react";
import Image from 'next/image';

// 支持的语言列表，从refer.md中提取
const SUPPORTED_LANGUAGES = [
  { value: '中文', label: '中文 (zh)' },
  { value: '英文', label: '英语 (en)' },
  { value: '法语', label: '法语 (fr)' },
  { value: '葡萄牙语', label: '葡萄牙语 (pt)' },
  { value: '西班牙语', label: '西班牙语 (es)' },
  { value: '日语', label: '日语 (ja)' },
  { value: '土耳其语', label: '土耳其语 (tr)' },
  { value: '俄语', label: '俄语 (ru)' },
  { value: '阿拉伯语', label: '阿拉伯语 (ar)' },
  { value: '韩语', label: '韩语 (ko)' },
  { value: '泰语', label: '泰语 (th)' },
  { value: '意大利语', label: '意大利语 (it)' },
  { value: '德语', label: '德语 (de)' },
  { value: '越南语', label: '越南语 (vi)' },
  { value: '马来语', label: '马来语 (ms)' },
  { value: '印尼语', label: '印尼语 (id)' },
  { value: '菲律宾语', label: '菲律宾语 (tl)' },
  { value: '印地语', label: '印地语 (hi)' },
  { value: '繁体中文', label: '繁体中文 (zh-Hant)' },
  { value: '波兰语', label: '波兰语 (pl)' },
  { value: '捷克语', label: '捷克语 (cs)' },
  { value: '荷兰语', label: '荷兰语 (nl)' },
  { value: '高棉语', label: '高棉语 (km)' },
  { value: '缅甸语', label: '缅甸语 (my)' },
  { value: '波斯语', label: '波斯语 (fa)' },
  { value: '古吉拉特语', label: '古吉拉特语 (gu)' },
  { value: '乌尔都语', label: '乌尔都语 (ur)' },
  { value: '泰卢固语', label: '泰卢固语 (te)' },
  { value: '马拉地语', label: '马拉地语 (mr)' },
  { value: '希伯来语', label: '希伯来语 (he)' },
  { value: '孟加拉语', label: '孟加拉语 (bn)' },
  { value: '泰米尔语', label: '泰米尔语 (ta)' },
  { value: '乌克兰语', label: '乌克兰语 (uk)' },
  { value: '藏语', label: '藏语 (bo)' },
  { value: '哈萨克语', label: '哈萨克语 (kk)' },
  { value: '蒙古语', label: '蒙古语 (mn)' },
  { value: '维吾尔语', label: '维吾尔语 (ug)' },
  { value: '粤语', label: '粤语 (yue)' }
];

export default function TranslatorApp() {
  const [sourceText, setSourceText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [sourceLang, setSourceLang] = useState('中文');
  const [targetLang, setTargetLang] = useState('英文');
  const [isLoading, setIsLoading] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const [error, setError] = useState('');
  const [endpoint, setEndpoint] = useState<'chat' | 'completion'>('chat');

  // 交换源语言和目标语言
  const swapLanguages = () => {
    const temp = sourceLang;
    setSourceLang(targetLang);
    setTargetLang(temp);
    // 如果已经有翻译结果，交换文本内容
    if (translatedText) {
      setSourceText(translatedText);
      setTranslatedText('');
    }
  };

  // 复制翻译结果到剪贴板
  const copyToClipboard = () => {
    if (translatedText) {
      navigator.clipboard.writeText(translatedText);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    }
  };

  // 执行翻译
  const handleTranslate = async () => {
    if (!sourceText.trim()) {
      setError('请输入要翻译的文本');
      setTimeout(() => setError(''), 3000);
      return;
    }

    setIsLoading(true);
    setError('');
    setTranslatedText('');

    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: sourceText,
          sourceLang,
          targetLang,
          endpoint
        })
      });

      const result = await response.json();

      if (result.success) {
        setTranslatedText(result.translatedText);
      } else {
        setError(result.error || '翻译失败，请稍后再试');
        setTimeout(() => setError(''), 5000);
      }
    } catch (err) {
      setError('网络错误，请检查连接');
      setTimeout(() => setError(''), 5000);
      console.error('翻译错误:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // 清空所有内容
  const clearAll = () => {
    setSourceText('');
    setTranslatedText('');
    setError('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-indigo-50 dark:from-gray-900 dark:to-indigo-950 flex flex-col">
      {/* 顶部导航栏 */}
      <header className="sticky top-0 z-10 backdrop-blur-lg bg-white/80 dark:bg-gray-900/80 border-b border-gray-200 dark:border-gray-800">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Language className="h-8 w-8 text-blue-600 dark:text-blue-400" />
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400 bg-clip-text text-transparent">
              25AEC Web翻译应用
            </h1>
          </div>
        </div>
      </header>

      {/* 主内容区域 */}
      <main className="flex-1 container mx-auto px-4 py-8 md:py-12">
        {/* 介绍部分 */}
        <section className="mb-8 text-center max-w-3xl mx-auto">
          <div className="mb-6 inline-flex justify-center">
            <Image 
              src="https://dscache.tencent-cloud.cn/upload/uploader/hunyuan-64b418fd052c033b228e04bc77bbc4b54fd7f5bc.png" 
              alt="Hunyuan MT Logo" 
              width={512} 
              height={256} 
            />
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-gray-800 dark:text-white">
            简易<span className="text-blue-600 dark:text-blue-400">多语言</span>翻译平台
          </h2>
          <p className="text-gray-600 dark:text-gray-300 mb-6 text-lg">
              基于自托管的vLLM腾讯混元翻译模型，支持<span className="font-semibold text-blue-600 dark:text-blue-400">33种语言</span>互译
            </p>
        </section>

        {/* 翻译区域 */}
        <section className="max-w-4xl mx-auto bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden transition-all duration-300 hover:shadow-2xl">
          {/* 语言选择区域 */}
          <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between bg-gray-50 dark:bg-gray-850">
            <div className="flex items-center gap-2">
              <Globe className="h-5 w-5 text-gray-500 dark:text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">模式和语言设置</span>
              <button 
              onClick={() => setEndpoint(endpoint === 'chat' ? 'completion' : 'chat')}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${endpoint === 'chat' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300' : 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900/50 dark:text-indigo-300'}`}
            >
              {endpoint === 'chat' ? 'Chat 模式' : 'Completion 模式'}
            </button>
            </div>
            <div className="flex items-center gap-4">
              <select
                value={sourceLang}
                onChange={(e) => setSourceLang(e.target.value)}
                className="text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {SUPPORTED_LANGUAGES.map((lang) => (
                  <option key={lang.value} value={lang.value}>
                    {lang.label}
                  </option>
                ))}
              </select>
              
              <button
                onClick={swapLanguages}
                disabled={isLoading}
                className="p-1.5 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 hover:bg-blue-200 dark:hover:bg-blue-800/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"              
                >
                  <RefreshCw className="h-4 w-4" />
              </button>
              
              <select
                value={targetLang}
                onChange={(e) => setTargetLang(e.target.value)}
                className="text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {SUPPORTED_LANGUAGES.map((lang) => (
                  <option key={lang.value} value={lang.value}>
                    {lang.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* 输入区域 */}
          <div className="p-6">
            <div className="mb-6">
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  输入文本
                </label>
                {sourceText && (
                  <button
                    onClick={clearAll}
                    className="text-xs text-gray-500 hover:text-red-500 dark:text-gray-400 dark:hover:text-red-400 transition-colors"
                  >
                    清空
                  </button>
                )}
              </div>
              <textarea
                value={sourceText}
                onChange={(e) => setSourceText(e.target.value)}
                placeholder={`请输入要翻译成${targetLang}的${sourceLang}文本...`}
                className="w-full min-h-[120px] p-4 border border-gray-300 dark:border-gray-600 rounded-xl bg-gray-50 dark:bg-gray-750 text-gray-800 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all resize-none"
                disabled={isLoading}
              />
            </div>

            {/* 翻译按钮 */}
            <div className="flex justify-center mb-6">
              <button
                onClick={handleTranslate}
                disabled={isLoading || !sourceText.trim()}
                className={`flex items-center gap-2 px-6 py-3 rounded-full font-medium text-white transition-all ${isLoading || !sourceText.trim() ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 hover:shadow-lg'}`}
              >
                {isLoading ? (
                  <Spinner className="h-4 w-5" variant="wave" />
                ) : (
                  <ArrowRight className="h-5 w-5" />
                )}
                {isLoading ? '翻译中...' : '开始翻译'}
              </button>
            </div>

            {/* 错误提示 */}
            {error && (
              <div className="mb-6 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-600 dark:text-red-400 text-sm flex items-center gap-2 animate-fade-in">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
                {error}
              </div>
            )}

            {/* 输出区域 */}
            {translatedText && (
              <div className="animate-fade-in">
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    翻译结果
                  </label>
                  <button
                    onClick={copyToClipboard}
                    className="p-1.5 rounded-full text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700 transition-colors"
                    aria-label="复制到剪贴板"
                  >
                    {copySuccess ? <Check className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
                  </button>
                </div>
                <div className="w-full min-h-[120px] p-4 border border-gray-300 dark:border-gray-600 rounded-xl bg-gray-50 dark:bg-gray-750 text-gray-800 dark:text-white">
                  {translatedText}
                </div>
              </div>
            )}
          </div>
        </section>
      </main>

      {/* 页脚 */}
      <footer className="mt-auto py-6 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800">
        <div className="container mx-auto px-4 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>© 2025 RainyHallways. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
