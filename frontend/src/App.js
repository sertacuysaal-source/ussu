import { useState } from "react";
import "@/App.css";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Loader2, TrendingUp, TrendingDown, Minus, Search } from "lucide-react";
import { toast } from "sonner";



const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("all");

  const handleScan = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/scan`, {});
      setSignals(response.data);
      toast.success(`${response.data.length} hisse tarandı`);
    } catch (error) {
      console.error("Scan error:", error);
      toast.error("Tarama sırasında hata oluştu");
    } finally {
      setLoading(false);
    }
  };

  const filterSignals = (signalType) => {
    if (signalType === "all") return signals;
    return signals.filter((s) => s.signal === signalType);
  };

  const getSignalBadgeColor = (signal) => {
    switch (signal) {
      case "GÜÇLÜ AL":
        return "bg-amber-500 text-white hover:bg-amber-600";
      case "AL":
        return "bg-emerald-500 text-white hover:bg-emerald-600";
      case "SAT":
        return "bg-rose-500 text-white hover:bg-rose-600";
      case "TUT":
        return "bg-slate-400 text-white hover:bg-slate-500";
      default:
        return "bg-gray-400 text-white";
    }
  };

  const getSignalIcon = (signal) => {
    switch (signal) {
      case "GÜÇLÜ AL":
      case "AL":
        return <TrendingUp className="w-4 h-4" />;
      case "SAT":
        return <TrendingDown className="w-4 h-4" />;
      case "TUT":
        return <Minus className="w-4 h-4" />;
      default:
        return null;
    }
  };

  const SignalCard = ({ signal }) => (
    <Card className="hover:shadow-lg transition-all duration-300 hover:-translate-y-1 border-l-4" 
          style={{
            borderLeftColor: 
              signal.signal === "GÜÇLÜ AL" ? "#f59e0b" :
              signal.signal === "AL" ? "#10b981" :
              signal.signal === "SAT" ? "#f43f5e" : "#94a3b8"
          }}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xl font-bold text-slate-800">
            {signal.symbol}
          </CardTitle>
          <Badge className={`${getSignalBadgeColor(signal.signal)} flex items-center gap-1`}>
            {getSignalIcon(signal.signal)}
            {signal.signal}
          </Badge>
        </div>
        <CardDescription className="flex items-center justify-between mt-2">
          <span className="text-2xl font-semibold text-slate-900">
            ₺{signal.price.toFixed(2)}
          </span>
          <span className={`text-sm font-medium ${
            signal.change_percent >= 0 ? "text-emerald-600" : "text-rose-600"
          }`}>
            {signal.change_percent >= 0 ? "+" : ""}{signal.change_percent?.toFixed(2)}%
          </span>
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-600">Sinyal Gücü:</span>
            <div className="flex items-center gap-2">
              <div className="w-24 h-2 bg-slate-200 rounded-full overflow-hidden">
                <div 
                  className="h-full transition-all duration-500"
                  style={{
                    width: `${signal.signal_strength}%`,
                    backgroundColor: 
                      signal.signal_strength >= 80 ? "#10b981" :
                      signal.signal_strength >= 60 ? "#f59e0b" : "#f43f5e"
                  }}
                />
              </div>
              <span className="font-semibold text-slate-700 w-8">{signal.signal_strength.toFixed(0)}</span>
            </div>
          </div>
          
          <div className="pt-2 border-t border-slate-200">
            <p className="text-xs font-medium text-slate-600 mb-2">Karşılanan Koşullar:</p>
            <div className="space-y-1">
              {signal.conditions_met.map((condition, idx) => (
                <div key={idx} className="flex items-start gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5 flex-shrink-0" />
                  <p className="text-xs text-slate-700 leading-relaxed">{condition}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );

  const signalCounts = {
    all: signals.length,
    "GÜÇLÜ AL": filterSignals("GÜÇLÜ AL").length,
    AL: filterSignals("AL").length,
    SAT: filterSignals("SAT").length,
    TUT: filterSignals("TUT").length,
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md shadow-sm sticky top-0 z-10 border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">

            {/* Logo + Başlık */}
            <div className="flex items-center gap-4">
              <img src={`${process.env.PUBLIC_URL}/logo.png`} alt="Site Logo" className="h-12 w-auto" />
              <div>
                <h1 className="text-4xl font-bold text-slate-900" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                  BIST Hisse Tarama Osmanlı Yatırım Uğur
                </h1>
                <p className="text-slate-600 mt-1 text-sm">Teknik analiz tabanlı otomatik sinyal sistemi</p>
              </div>
            </div>

            {/* Tarama Butonu */}
            <Button 
              onClick={handleScan} 
              disabled={loading}
              size="lg"
              className="bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 text-white font-semibold shadow-lg hover:shadow-xl transition-all duration-300"
              data-testid="scan-button"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Taranıyor...
                </>
              ) : (
                <>
                  <Search className="mr-2 h-5 w-5" />
                  Hisseleri Tara
                </>
              )}
            </Button>

          </div>
        </div>
      </header>


      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {signals.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20" data-testid="empty-state">
            <div className="w-24 h-24 bg-gradient-to-br from-indigo-100 to-blue-100 rounded-full flex items-center justify-center mb-6">
              <Search className="w-12 h-12 text-indigo-600" />
            </div>
            <h2 className="text-2xl font-semibold text-slate-800 mb-2">Tarama Başlatın</h2>
            <p className="text-slate-600 text-center max-w-md">
              BIST hisselerini teknik indikatörlerle analiz etmek için "Hisseleri Tara" butonuna tıklayın.
            </p>
          </div>
        ) : (
          <Tabs defaultValue="all" className="w-full" onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-5 mb-8 bg-white p-1 shadow-sm" data-testid="signal-tabs">
              <TabsTrigger value="all" className="data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all">
                Tümü ({signalCounts.all})
              </TabsTrigger>
              <TabsTrigger value="GÜÇLÜ AL" className="data-[state=active]:bg-amber-500 data-[state=active]:text-white transition-all">
                Güçlü Al ({signalCounts["GÜÇLÜ AL"]})
              </TabsTrigger>
              <TabsTrigger value="AL" className="data-[state=active]:bg-emerald-500 data-[state=active]:text-white transition-all">
                Al ({signalCounts.AL})
              </TabsTrigger>
              <TabsTrigger value="SAT" className="data-[state=active]:bg-rose-500 data-[state=active]:text-white transition-all">
                Sat ({signalCounts.SAT})
              </TabsTrigger>
              <TabsTrigger value="TUT" className="data-[state=active]:bg-slate-400 data-[state=active]:text-white transition-all">
                Tut ({signalCounts.TUT})
              </TabsTrigger>
            </TabsList>

            {["all", "GÜÇLÜ AL", "AL", "SAT", "TUT"].map((tab) => (
              <TabsContent key={tab} value={tab} className="mt-0">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6" data-testid={`${tab}-signals-grid`}>
                  {filterSignals(tab).map((signal, idx) => (
                    <SignalCard key={idx} signal={signal} />
                  ))}
                </div>
                {filterSignals(tab).length === 0 && (
                  <div className="text-center py-12 text-slate-500">
                    Bu kategoride sinyal bulunamadı.
                  </div>
                )}
              </TabsContent>
            ))}
          </Tabs>
        )}
      </main>
    </div>
  );
}

export default App;
