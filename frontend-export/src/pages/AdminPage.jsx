
import React from 'react';
import { motion } from 'framer-motion';
import { ShieldCheck, Users, BarChart2, Settings } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { useToast } from '@/components/ui/use-toast';
import { Button } from '@/components/ui/button';

const AdminCard = ({ title, description, icon: Icon, actionText }) => {
    const { toast } = useToast();
    const handleAction = () => {
        toast({
            title: '🚧 Feature in progress',
            description: `The "${title}" feature isn't implemented yet.`,
        });
    };

    return (
        <motion.div whileHover={{ y: -5, scale: 1.02 }} transition={{ type: 'spring', stiffness: 300 }}>
            <Card className="bg-slate-800/50 border-slate-700/50 h-full flex flex-col">
                <CardHeader className="flex flex-row items-center gap-4">
                    <div className="p-3 bg-purple-500/20 rounded-lg">
                        <Icon className="w-6 h-6 text-purple-400" />
                    </div>
                    <div>
                        <CardTitle>{title}</CardTitle>
                        <CardDescription className="text-slate-400">{description}</CardDescription>
                    </div>
                </CardHeader>
                <CardContent className="flex-grow flex items-end">
                    <Button onClick={handleAction} variant="secondary" className="w-full bg-slate-700 hover:bg-slate-600">
                        {actionText}
                    </Button>
                </CardContent>
            </Card>
        </motion.div>
    );
};

export function AdminPage() {
  return (
    <div className="space-y-8">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <h1 className="text-4xl font-bold tracking-tighter text-white flex items-center gap-3">
            <ShieldCheck className="w-10 h-10 text-purple-400"/>
            Admin Dashboard
        </h1>
        <p className="text-slate-400 mt-2">Manage users, view analytics, and configure application settings.</p>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <AdminCard 
            title="User Management"
            description="View, edit, and manage all users."
            icon={Users}
            actionText="Manage Users"
        />
        <AdminCard 
            title="Analytics"
            description="Track song generation and user activity."
            icon={BarChart2}
            actionText="View Analytics"
        />
        <AdminCard 
            title="App Settings"
            description="Configure global application settings."
            icon={Settings}
            actionText="Configure Settings"
        />
      </div>
    </div>
  );
}
